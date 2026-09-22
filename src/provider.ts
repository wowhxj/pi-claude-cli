/**
 * Provider orchestration for bridging pi requests to the Claude CLI subprocess.
 *
 * streamViaCli is the core function that:
 * 1. Builds the prompt from conversation context
 * 2. Spawns a Claude CLI subprocess with correct flags
 * 3. Writes the user message to stdin as NDJSON
 * 4. Reads stdout line-by-line, parsing NDJSON
 * 5. Routes stream events through the event bridge to pi's stream
 * 6. Handles result/error messages and cleans up the subprocess
 * 7. Implements break-early: kills subprocess at message_stop when
 *    built-in or custom-tools MCP tool_use blocks are seen
 * 8. Hardened lifecycle: inactivity timeout, subprocess exit handler,
 *    streamEnded guard, abort via SIGKILL, process registry
 */

import { randomUUID } from "node:crypto";
import { createInterface } from "node:readline";
import {
  AssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import {
  buildPrompt,
  buildSystemPrompt,
  buildResumePrompt,
} from "./prompt-builder.js";
import {
  spawnClaude,
  writeUserMessage,
  cleanupProcess,
  captureStderr,
  forceKillProcess,
  registerProcess,
  cleanupSystemPromptFile,
} from "./process-manager.js";
import { parseLine } from "./stream-parser.js";
import { createEventBridge } from "./event-bridge.js";
import { handleControlRequest } from "./control-handler.js";
import { mapThinkingEffort } from "./thinking-config.js";
import { isPiKnownClaudeTool } from "./tool-mapping.js";

export const CLAUDE_AUTO_COMPACT_THRESHOLD_CHARS = 64_000;

const PI_COMPACTION_SUMMARY_PREFIX =
  "The conversation history before this point was compacted into the following summary:";

interface ClaudeSessionState {
  sessionId: string;
  compactionMarker?: string;
}

/**
 * Maps a Pi session/model pair to the Claude CLI session currently backing it.
 *
 * Normally both IDs are the same. After Pi compacts its context, the old
 * Claude conversation cannot be compacted by this provider, so a fresh Claude
 * session is created and remembered here for subsequent --resume calls.
 */
const claudeSessionStates = new Map<string, ClaudeSessionState>();
/** Inactivity timeout: kill subprocess if no stdout for 180 seconds (3 minutes). */
const INACTIVITY_TIMEOUT_MS = 180_000;

/**
 * Return true when a message contains a usable assistant response.
 * Empty assistant messages are how pi records a failed provider turn.
 */
function hasMessageContent(content: unknown): boolean {
  if (typeof content === "string") return content.trim().length > 0;
  return Array.isArray(content) && content.length > 0;
}

/**
 * Resume only a Claude CLI session that this provider actually created.
 *
 * Pi prepends a system message before calling providers, so message-count
 * heuristics such as `messages.length > 1` mistake the first [system, user]
 * turn for a follow-up. Provider/model matching also prevents a Codex turn
 * from being used as evidence that a Claude CLI session exists.
 */
export function hasPriorClaudeTurn(messages: any[], modelId: string): boolean {
  return messages.some(
    (message) =>
      message?.role === "assistant" &&
      (message.api === "pi-claude-cli" ||
        message.provider === "pi-claude-cli") &&
      message.model === modelId &&
      hasMessageContent(message.content),
  );
}

function messageContentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .filter((block) => block?.type === "text")
    .map((block) => block.text ?? "")
    .join("\n");
}

/**
 * Return a stable marker for the latest Pi compaction summary, if present.
 * Pi converts compactionSummary messages to user messages before providers see
 * them, so detect the prefix emitted by pi's message transformer as well as
 * the raw custom role for compatibility with future Pi versions.
 */
export function hasCompactionSummary(messages: any[]): boolean {
  return getCompactionMarker(messages) !== undefined;
}

export function estimateContextChars(messages: any[]): number {
  try {
    return JSON.stringify(messages).length;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

function getCompactionMarker(messages: any[]): string | undefined {
  let marker: string | undefined;

  for (const message of messages) {
    if (message?.role === "compactionSummary") {
      marker = JSON.stringify(message);
      continue;
    }

    if (message?.role !== "user") continue;
    const text = messageContentToText(message.content);
    if (text.startsWith(PI_COMPACTION_SUMMARY_PREFIX)) {
      marker = text;
    }
  }

  return marker;
}

function getSessionKey(
  sessionId: string | undefined,
  modelId: string,
): string | undefined {
  return sessionId ? `${sessionId}\u0000${modelId}` : undefined;
}

function getClaudeSessionPlan(
  messages: any[],
  modelId: string,
  piSessionId: string | undefined,
): {
  resumeSessionId?: string;
  newSessionId?: string;
} {
  const key = getSessionKey(piSessionId, modelId);
  const marker = getCompactionMarker(messages);
  const state = key ? claudeSessionStates.get(key) : undefined;

  if (marker) {
    // A new marker means Pi compacted again. Start a new Claude conversation
    // so the old, un-compacted Claude transcript is not retained forever.
    if (!state || state.compactionMarker !== marker) {
      const newSessionId = hasPriorClaudeTurn(messages, modelId)
        ? randomUUID()
        : (piSessionId ?? randomUUID());

      if (key) {
        claudeSessionStates.set(key, {
          sessionId: newSessionId,
          compactionMarker: marker,
        });
      }

      return { newSessionId };
    }

    return { resumeSessionId: state.sessionId };
  }

  if (hasPriorClaudeTurn(messages, modelId)) {
    return {
      resumeSessionId: state?.sessionId ?? piSessionId,
    };
  }

  const newSessionId = piSessionId;
  if (key && newSessionId) {
    claudeSessionStates.set(key, { sessionId: newSessionId });
  }
  return { newSessionId };
}

function getResultErrorMessage(message: {
  subtype?: string;
  error?: string;
  errors?: string[];
  result?: string;
}): string {
  const details = [
    message.error,
    ...(Array.isArray(message.errors) ? message.errors : []),
  ].filter((value): value is string => Boolean(value?.trim()));

  return (
    details.join("; ") ||
    message.result ||
    `Claude CLI returned ${message.subtype ?? "an error"}`
  );
}

/** Extended stream options: pi's SimpleStreamOptions plus optional cwd and mcpConfigPath */
type StreamViaCLiOptions = SimpleStreamOptions & {
  cwd?: string;
  mcpConfigPath?: string;
};

/**
 * Stream a response from Claude CLI as an AssistantMessageEventStream.
 *
 * Orchestrates the full subprocess lifecycle: spawn, write prompt, parse NDJSON,
 * bridge events, handle result, and clean up. Implements break-early pattern:
 * at message_stop, if any built-in or custom-tools MCP tool was seen, kills
 * the subprocess before Claude CLI can auto-execute the tools.
 *
 * Hardened with: inactivity timeout (180s), subprocess exit handler with stderr
 * surfacing, streamEnded guard against double errors, abort via SIGKILL, and
 * process registry integration for teardown cleanup.
 *
 * @param model - The model to use (from pi's model catalog)
 * @param context - The conversation context with messages and system prompt
 * @param options - Optional cwd, abort signal, reasoning level, thinking budgets, and mcpConfigPath
 * @returns An AssistantMessageEventStream that receives bridged events
 */
export function streamViaCli(
  model: Model<any>,
  context: { messages: any[]; systemPrompt?: string },
  options?: StreamViaCLiOptions,
): AssistantMessageEventStream {
  // @ts-expect-error — tsc can't verify AssistantMessageEventStream is a value
  // through pi-ai's `export *` re-export chain. The class constructor exists at runtime.
  const stream = new AssistantMessageEventStream();

  (async () => {
    let proc: ReturnType<typeof spawnClaude> | undefined;
    let abortHandler: (() => void) | undefined;

    try {
      const cwd = options?.cwd ?? process.cwd();

      // Pi passes sessionId on every call, including the first turn. Only
      // resume after a successful response from this exact Claude model.
      // The provider-facing transcript contains a leading system message, so
      // context.messages.length is not a reliable first-turn check.
      // Pi uses cacheRetention: "none" for one-off compaction summaries.
      // Do not attach those requests to the persistent Claude session: an
      // automatic pre-Claude compaction must not consume the same session ID
      // that the real post-compaction request is about to initialize.
      const { resumeSessionId, newSessionId } =
        options?.cacheRetention === "none"
          ? {}
          : getClaudeSessionPlan(
              context.messages,
              model.id,
              options?.sessionId,
            );

      // Build prompt: if resuming, only send the latest user turn;
      // otherwise build the full flattened conversation history. A Pi
      // compaction always takes the new-session path, so the summary and the
      // messages retained after it become Claude's initial context.
      const prompt = resumeSessionId
        ? buildResumePrompt(context)
        : buildPrompt(context);
      const systemPrompt = resumeSessionId
        ? undefined
        : buildSystemPrompt(context, cwd);

      // Compute effort level from reasoning options
      const effort = mapThinkingEffort(
        options?.reasoning,
        model.id,
        options?.thinkingBudgets,
      );

      // Spawn subprocess
      proc = spawnClaude(model.id, systemPrompt || undefined, {
        cwd,
        signal: options?.signal,
        effort,
        mcpConfigPath: options?.mcpConfigPath,
        resumeSessionId,
        newSessionId,
      });
      const getStderr = captureStderr(proc);

      // Register in global process registry for teardown cleanup
      registerProcess(proc);

      // Write user message to subprocess stdin
      writeUserMessage(proc, prompt);

      // Create event bridge (before endStreamWithError so bridge is in scope)
      const bridge = createEventBridge(stream, model);

      // Guard against double stream.end() and double error events.
      // First error path wins; subsequent ones are no-ops.
      let streamEnded = false;

      /**
       * End the stream with an error, using a "done" event instead of "error".
       *
       * Why "done" not "error": AssistantMessageEventStream.extractResult()
       * returns event.error (a string) for error events, but agent-loop.js
       * then calls message.content.filter() on the result, crashing because
       * a string has no .content property. By pushing "done" with a valid
       * AssistantMessage (content:[]), pi gets a well-formed object.
       */
      function endStreamWithError(errMsg: string) {
        if (streamEnded || broken) return;
        streamEnded = true;
        const output = bridge.getOutput();
        const errorMessage = {
          ...output,
          content: output.content?.length
            ? output.content
            : [{ type: "text" as const, text: `Error: ${errMsg}` }],
          stopReason: "stop" as const,
        };
        stream.push({
          type: "done",
          reason: "stop",
          message: errorMessage,
        } as any);
        stream.end();
      }

      // Inactivity timeout: kill subprocess if no stdout for INACTIVITY_TIMEOUT_MS
      let inactivityTimer: ReturnType<typeof setTimeout> | undefined;

      function resetInactivityTimer() {
        if (inactivityTimer !== undefined) clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(() => {
          forceKillProcess(proc!);
          endStreamWithError(
            `Claude CLI subprocess timed out: no output for ${INACTIVITY_TIMEOUT_MS / 1000} seconds`,
          );
        }, INACTIVITY_TIMEOUT_MS);
      }

      // Set up abort signal handler -- uses SIGKILL for immediate force-kill
      if (options?.signal) {
        abortHandler = () => {
          if (proc) {
            forceKillProcess(proc);
          }
        };

        if (options.signal.aborted) {
          abortHandler();
          return;
        }
        options.signal.addEventListener("abort", abortHandler, { once: true });
      }

      // Track tool_use blocks for break-early decision at message_stop
      let sawBuiltInOrCustomTool = false;
      // Guard against buffered readline lines firing after rl.close()
      let broken = false;

      // Set up readline for line-by-line NDJSON parsing
      const rl = createInterface({
        input: proc.stdout!,
        crlfDelay: Infinity,
        terminal: false,
      });

      // Handle process error -- use endStreamWithError for guard
      proc.on("error", (err: Error) => {
        if (broken) return; // Break-early killed the process intentionally
        const stderr = getStderr();
        endStreamWithError(stderr || err.message);
      });

      // Handle subprocess close -- surface crashes with stderr and exit code
      proc.on("close", (code: number | null, _signal: string | null) => {
        clearTimeout(inactivityTimer);
        if (broken) return; // Break-early kill, expected
        if (code !== 0 && code !== null) {
          const stderr = getStderr();
          const message = stderr
            ? `Claude CLI exited with code ${code}: ${stderr.trim()}`
            : `Claude CLI exited unexpectedly with code ${code}`;
          endStreamWithError(message);
        }
      });

      // Start inactivity timer after writing user message
      resetInactivityTimer();

      // Process NDJSON lines from stdout using event-based callback
      // NOTE: Using 'line' event instead of `for await` because the async
      // iterator batches lines, breaking real-time streaming to pi.
      rl.on("line", (line: string) => {
        if (broken) return; // Guard: ignore buffered lines after break-early

        // Reset inactivity timer on each line of output
        resetInactivityTimer();

        const msg = parseLine(line);
        if (!msg) return;

        if (msg.type === "stream_event") {
          // Only forward top-level events to pi's event bridge.
          // Sub-agent events (parent_tool_use_id !== null) are internal to the CLI.
          const isTopLevel = !(msg as any).parent_tool_use_id;
          if (isTopLevel) {
            bridge.handleEvent(msg.event);
          }

          // Track tool_use blocks for break-early decision (top-level only)
          if (
            isTopLevel &&
            msg.event.type === "content_block_start" &&
            msg.event.content_block?.type === "tool_use"
          ) {
            const toolName = msg.event.content_block.name;
            if (toolName && isPiKnownClaudeTool(toolName)) {
              // Built-in tool (Read/Write/etc.) OR custom MCP tool (mcp__custom-tools__*)
              // Internal Claude Code tools (ToolSearch, Task, etc.) are excluded
              sawBuiltInOrCustomTool = true;
            }
          }

          // Break-early at message_stop: kill subprocess before CLI auto-executes tools
          // Only on top-level message_stop — sub-agent message_stop is internal
          if (
            isTopLevel &&
            msg.event.type === "message_stop" &&
            sawBuiltInOrCustomTool
          ) {
            broken = true; // Set guard BEFORE rl.close() to prevent buffered lines
            clearTimeout(inactivityTimer);
            // Pi will execute these tools. Kill subprocess to prevent CLI from executing them.
            forceKillProcess(proc!);
            rl.close();
            return; // Don't process further -- done event already pushed by event bridge
          }
        } else if (msg.type === "control_request") {
          handleControlRequest(msg, proc!.stdin!);
        } else if (msg.type === "result") {
          // Claude CLI 2.x uses several non-success subtypes, including
          // error_during_execution. Treat every subtype other than success as
          // an error; otherwise pi receives an empty successful response.
          if (msg.subtype !== "success") {
            endStreamWithError(getResultErrorMessage(msg));
          }
          // For both success and error: clean up the subprocess
          clearTimeout(inactivityTimer);
          cleanupProcess(proc!);
          rl.close();
        }
      });

      // Wait for readline to close (result received or process ended)
      await new Promise<void>((resolve) => {
        rl.on("close", resolve);
      });

      // Push done event after readline closes (async). Pushing synchronously
      // inside handleMessageStop prevents pi from executing tools.
      // Guard with streamEnded to avoid pushing done after an error was already pushed.
      if (!streamEnded) {
        const output = bridge.getOutput();

        // If stopReason is toolUse but there are no pi-known tool calls in content,
        // it means only user MCP tools were called (filtered by event bridge).
        // Override to "stop" so pi doesn't try to execute non-existent tools.
        const piToolCalls = (output.content || []).filter(
          (c: any) => c.type === "toolCall",
        );
        const effectiveReason =
          output.stopReason === "toolUse" && piToolCalls.length === 0
            ? "stop"
            : output.stopReason;

        streamEnded = true;
        stream.push({
          type: "done",
          reason:
            effectiveReason === "toolUse"
              ? "toolUse"
              : effectiveReason === "length"
                ? "length"
                : "stop",
          message: { ...output, stopReason: effectiveReason },
        });
        stream.end();
      }
    } catch (err: any) {
      stream.push({
        type: "error",
        reason: "error",
        error: err.message ?? "Unexpected error in streamViaCli",
      } as any);
      stream.end();
    } finally {
      // Clean up abort listener
      if (options?.signal && abortHandler) {
        options.signal.removeEventListener("abort", abortHandler);
      }
      cleanupSystemPromptFile();
    }
  })();

  return stream;
}
