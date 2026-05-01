import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { Panel, PanelContent, PanelHeader, PanelTitle } from "@pipecat-ai/voice-ui-kit";
import { useEffect, useRef, useState } from "react";

type BashToolResult = {
  ok?: boolean;
  exit_code?: number | null;
  stdout?: string;
  stderr?: string;
  timed_out?: boolean;
  elapsed_secs?: number;
  cwd?: string;
  error?: string;
};

type BashToolMessage = {
  type?: string;
  phase?: "start" | "result";
  tool_call_id?: string;
  code?: string;
  cwd?: string;
  result?: BashToolResult;
};

type TerminalEntry = {
  id: string;
  code: string;
  cwd?: string;
  status: "running" | "done";
  result?: BashToolResult;
};

const isBashToolMessage = (message: unknown): message is BashToolMessage => {
  if (!message || typeof message !== "object") return false;
  const value = message as BashToolMessage;
  return value.type === "bash-tool";
};

const normalizeOutput = (text: string | undefined) => {
  if (text === undefined || text === "") return "";
  return text.endsWith("\n") ? text : `${text}\n`;
};

export const BashTerminalPanel = () => {
  const [entries, setEntries] = useState<TerminalEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useRTVIClientEvent(RTVIEvent.ServerMessage, (message: unknown) => {
    if (!isBashToolMessage(message)) return;

    const id = message.tool_call_id || `call-${Date.now()}`;
    const code = message.code || "";

    if (message.phase === "start") {
      setEntries((current) => [
        ...current,
        {
          id,
          code,
          cwd: message.cwd,
          status: "running",
        },
      ]);
      return;
    }

    if (message.phase === "result") {
      setEntries((current) => {
        const index = current.findIndex((entry) => entry.id === id);
        if (index === -1) {
          return [
            ...current,
            {
              id,
              code,
              cwd: message.cwd,
              status: "done",
              result: message.result,
            },
          ];
        }

        const next = [...current];
        next[index] = {
          ...next[index],
          code: next[index].code || code,
          cwd: next[index].cwd || message.cwd,
          status: "done",
          result: message.result,
        };
        return next;
      });
    }
  });

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [entries]);

  return (
    <Panel className="h-full min-h-0 rounded-none! sm:border-x-0 sm:border-b-0">
      <PanelHeader className="justify-start bg-background">
        <PanelTitle>Terminal</PanelTitle>
      </PanelHeader>
      <PanelContent
        ref={scrollRef}
        className="terminal-panel min-h-0 flex-1 overflow-y-auto p-0!"
      >
        <div className="terminal-screen">
          {entries.length === 0 ? (
            <div className="terminal-cursor">$</div>
          ) : (
            entries.map((entry, index) => {
              const stdout = normalizeOutput(entry.result?.stdout);
              const stderr = normalizeOutput(entry.result?.stderr);
              const error = entry.result?.error;
              return (
                <div className="terminal-entry" key={`${entry.id}-${index}`}>
                  <div className="terminal-command">
                    <span className="terminal-prompt">$</span>
                    <span>{entry.code}</span>
                  </div>
                  {stdout && <pre className="terminal-output">{stdout}</pre>}
                  {stderr && <pre className="terminal-error">{stderr}</pre>}
                  {error && <pre className="terminal-error">{error}</pre>}
                  {entry.status === "running" ? (
                    <div className="terminal-status">running...</div>
                  ) : (
                    <div className="terminal-status">
                      exit {entry.result?.exit_code ?? "?"}
                      {entry.result?.timed_out ? " timed out" : ""}
                      {typeof entry.result?.elapsed_secs === "number"
                        ? ` ${entry.result.elapsed_secs}s`
                        : ""}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </PanelContent>
    </Panel>
  );
};
