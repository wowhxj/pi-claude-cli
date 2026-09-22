/**
 * Pi extension entry point for pi-claude-cli.
 *
 * Registers a custom provider that routes LLM calls through the Claude Code CLI
 * subprocess using stream-json NDJSON protocol.
 */

import { getModels } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
  CLAUDE_AUTO_COMPACT_THRESHOLD_CHARS,
  estimateContextChars,
  hasCompactionSummary,
  hasPriorClaudeTurn,
  streamViaCli,
} from "./src/provider.js";
import {
  validateCliPresence,
  validateCliAuth,
  killAllProcesses,
} from "./src/process-manager.js";
import { getCustomToolDefs, writeMcpConfig } from "./src/mcp-config.js";

// Kill all active Claude subprocesses on process exit to prevent orphans
process.on("exit", killAllProcesses);

const PROVIDER_ID = "pi-claude-cli";

let mcpConfigPath: string | undefined;
let mcpConfigResolved = false;
let autoCompactionInProgress = false;

/**
 * Lazily generate MCP config on first request (not at load time).
 * pi.getAllTools() fails during extension loading; this defers it
 * until the pi runtime is fully initialized.
 *
 * Only locks (sets mcpConfigResolved) when getAllTools() returns a
 * real array — if it returns undefined/null (registry not ready),
 * we retry on the next request. Once the registry is ready we
 * commit to the result even if there are zero custom tools.
 *
 * Uses warn-don't-block: failure logs a warning but does not
 * prevent the provider from functioning (built-ins still work).
 */
function ensureMcpConfig(pi: ExtensionAPI): string | undefined {
  if (mcpConfigResolved) return mcpConfigPath;
  try {
    const allTools = pi.getAllTools();

    // Registry not ready yet — don't lock, retry on next call
    if (!Array.isArray(allTools)) {
      return mcpConfigPath;
    }

    // Registry is ready — lock regardless of whether custom tools exist
    mcpConfigResolved = true;

    const toolDefs = getCustomToolDefs(pi);
    if (toolDefs.length > 0) {
      mcpConfigPath = writeMcpConfig(toolDefs);
      console.error(
        `[pi-claude-cli] MCP config generated with ${toolDefs.length} custom tool(s)`,
      );
    }
  } catch (err) {
    console.warn(
      "[pi-claude-cli] MCP config generation failed, custom tools unavailable:",
      err,
    );
  }
  return mcpConfigPath;
}

export default function (pi: ExtensionAPI) {
  try {
    // Startup validation
    validateCliPresence(); // throws if CLI not on PATH
    validateCliAuth(); // warns if not authenticated

    const models = getModels("anthropic").map((model) => ({
      id: model.id,
      name: model.name,
      reasoning: model.reasoning,
      input: model.input,
      cost: model.cost,
      contextWindow: model.contextWindow,
      maxTokens: model.maxTokens,
    }));

    // Ensure all registered tools are active so pi can execute them.
    // Some tools (find, grep, ls) are registered but not activated by default.
    pi.on("session_start", async () => {
      const allTools = pi.getAllTools();
      if (Array.isArray(allTools)) {
        pi.setActiveTools(allTools.map((t: any) => t.name));
      }
    });

    // A model switch into Claude otherwise sends the entire Pi history as the
    // first Claude prompt. Compact that history before the provider request
    // when it is large enough to make the first request expensive.
    pi.on("context", async (event, ctx) => {
      const model = ctx.model;
      if (
        autoCompactionInProgress ||
        model?.provider !== PROVIDER_ID ||
        hasPriorClaudeTurn(event.messages, model.id) ||
        hasCompactionSummary(event.messages) ||
        estimateContextChars(event.messages) <=
          CLAUDE_AUTO_COMPACT_THRESHOLD_CHARS
      ) {
        return;
      }

      autoCompactionInProgress = true;
      console.error(
        `[pi-claude-cli] Compacting ${estimateContextChars(event.messages).toLocaleString()} characters before the first Claude request`,
      );

      try {
        await new Promise<void>((resolve, reject) => {
          ctx.compact({
            onComplete: () => resolve(),
            onError: reject,
          });
        });

        // compact() updates the append-only session. Return its fresh,
        // compaction-aware messages to the current LLM turn instead of the
        // pre-compaction snapshot that triggered this handler.
        const refreshedContext = (
          ctx.sessionManager as typeof ctx.sessionManager & {
            buildSessionContext?: () => { messages: any[] };
          }
        ).buildSessionContext?.();

        if (refreshedContext?.messages) {
          return { messages: refreshedContext.messages };
        }

        console.warn(
          "[pi-claude-cli] Pi did not expose the refreshed compacted context; continuing with the original context",
        );
        return;
      } catch (error) {
        console.warn(
          "[pi-claude-cli] Automatic pre-Claude compaction failed; continuing with the original context:",
          error,
        );
        return;
      } finally {
        autoCompactionInProgress = false;
      }
    });

    pi.registerProvider(PROVIDER_ID, {
      baseUrl: "pi-claude-cli",
      apiKey: "unused",
      api: "pi-claude-cli",
      models,
      streamSimple: (model, context, options) => {
        const configPath = ensureMcpConfig(pi);
        return streamViaCli(model, context, {
          ...options,
          mcpConfigPath: configPath,
        });
      },
    });
  } catch (err) {
    console.error(`[pi-claude-cli] Failed to register provider:`, err);
  }
}
