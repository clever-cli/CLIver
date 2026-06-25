import { useCallback, useEffect, useRef, useState } from "react";
import { WsMessageSchema, type WsMessage } from "@/lib/schemas";

export type { WsMessage };

export function useWebSocket(
  url: string | null,
  onMessage?: (data: WsMessage) => void,
) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!url) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setIsConnected(false);
      }
      return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${url}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;
    };
    ws.onerror = () => {
      setIsConnected(false);
    };
    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        const data = WsMessageSchema.parse(raw);
        setLastMessage(data);
        // Call onMessage synchronously from the event handler to avoid
        // React batching dropping intermediate messages.
        onMessageRef.current?.(data);
      } catch {
        // ignore non-JSON or invalid messages
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const close = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setIsConnected(false);
  }, []);

  return { isConnected, lastMessage, send, close };
}
