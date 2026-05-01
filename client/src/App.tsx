import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import {
  Banner,
  BannerTitle,
  BotAudioPanel,
  ConnectButton,
  ConversationPanel,
  EventsPanel,
  FullScreenContainer,
  InfoPanel,
  PipecatAppBase,
  PipecatLogo,
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
  SpinLoader,
  ThemeModeToggle,
  ThemeProvider,
  type PipecatBaseChildProps,
} from "@pipecat-ai/voice-ui-kit";
import { useEffect, useMemo, useState } from "react";
import { BashTerminalPanel } from "./BashTerminalPanel";

const startBotParams = {
  endpoint: "/start",
  requestData: {
    createDailyRoom: false,
    enableDefaultIceServers: true,
    transport: "webrtc",
  },
};

const transportOptions = {
  waitForICEGathering: true,
};

const sessionIdFromResponse = (response: unknown) => {
  if (!response || typeof response !== "object") return "";
  const sessionId = (response as { sessionId?: unknown }).sessionId;
  return typeof sessionId === "string" ? sessionId : "";
};

const ConsoleSurface = ({
  client,
  error,
  handleConnect,
  handleDisconnect,
  rawStartBotResponse,
}: PipecatBaseChildProps) => {
  const [participantId, setParticipantId] = useState("");
  const [botStartedSessionId, setBotStartedSessionId] = useState("");

  const sessionId = useMemo(
    () => botStartedSessionId || sessionIdFromResponse(rawStartBotResponse),
    [botStartedSessionId, rawStartBotResponse],
  );

  useRTVIClientEvent(RTVIEvent.ParticipantConnected, (participant) => {
    if (participant.local) setParticipantId(participant.id || "");
  });

  useRTVIClientEvent(RTVIEvent.TrackStarted, (_track, participant) => {
    if (participant?.local && participant.id) setParticipantId(participant.id);
  });

  useRTVIClientEvent(RTVIEvent.BotStarted, (data) => {
    const sessionId = (data as { sessionId?: unknown })?.sessionId;
    if (typeof sessionId === "string") setBotStartedSessionId(sessionId);
  });

  useEffect(() => {
    if (!client) {
      setParticipantId("");
      setBotStartedSessionId("");
    }
  }, [client]);

  if (!client) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <SpinLoader />
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden">
      <header className="grid h-min grid-cols-2 items-center justify-center gap-2 bg-background p-2 sm:grid-cols-[150px_1fr_150px]">
        <PipecatLogo className="h-6 w-auto text-foreground" />
        <strong className="hidden text-center sm:block">
          Pipecat Playground
        </strong>
        <div className="flex items-center justify-end gap-2 sm:gap-3">
          <ThemeModeToggle />
          <ConnectButton
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
          />
        </div>
      </header>

      {error && (
        <Banner variant="destructive" className="h-min animate-in fade-in duration-300">
          <BannerTitle>
            Unable to connect. Please check web console for errors.
          </BannerTitle>
        </Banner>
      )}

      <div className="hidden min-h-0 flex-1 sm:block">
        <ResizablePanelGroup direction="vertical" className="h-full">
          <ResizablePanel defaultSize={50} minSize={35}>
            <ResizablePanelGroup direction="horizontal">
              <ResizablePanel
                className="flex min-h-0 flex-col gap-2 p-2 xl:gap-4"
                defaultSize={26}
                maxSize={34}
                minSize={16}
              >
                <BotAudioPanel className="bot-audio-panel" />
                <div className="rtvi-events-panel min-h-0 flex-1">
                  <EventsPanel />
                </div>
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel className="h-full p-2" defaultSize={47} minSize={30}>
                <ConversationPanel
                  conversationElementProps={{
                    assistantLabel: "assistant",
                    clientLabel: "user",
                    systemLabel: "system",
                  }}
                />
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel className="p-2" defaultSize={27} minSize={15}>
                <InfoPanel
                  noUserVideo
                  noScreenControl
                  participantId={participantId}
                  sessionId={sessionId}
                />
              </ResizablePanel>
            </ResizablePanelGroup>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={50} minSize={12}>
            <BashTerminalPanel />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      <div className="grid min-h-0 flex-1 grid-rows-[auto_1fr_1fr_1fr] gap-2 overflow-y-auto p-2 sm:hidden">
        <BotAudioPanel />
        <ConversationPanel />
        <div className="rtvi-events-panel min-h-0">
          <EventsPanel />
        </div>
        <BashTerminalPanel />
      </div>
    </div>
  );
};

export const App = () => {
  return (
    <ThemeProvider>
      <FullScreenContainer>
        <PipecatAppBase
          startBotParams={startBotParams}
          transportType="smallwebrtc"
          transportOptions={transportOptions}
          noThemeProvider
          initDevicesOnMount
        >
          {(props: PipecatBaseChildProps) => <ConsoleSurface {...props} />}
        </PipecatAppBase>
      </FullScreenContainer>
    </ThemeProvider>
  );
};
