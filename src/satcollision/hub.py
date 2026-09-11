#!/usr/bin/env python3
"""
The federation's network layer: a real asyncio TCP server that relays
messages between operator processes.

This is deliberately a "dumb pipe." It never parses a message beyond the
outer length-prefixed frame (``wire.py``), never checks a signature, and
never decides whether a message is legitimate -- that is entirely
``operator_node.py``'s job on each receiving end, exactly as it would be
on a real network where the wire carrying a packet has no idea what's
inside it. Any message a client sends is broadcast verbatim to every
*other* client, and also kept in an in-memory history replayed to every
new client as soon as it connects -- so delivery of any given message to
any given client happens exactly once, regardless of connection order,
closer to a durable message bus than a live-only broadcast. This is what
makes the security property demonstrated in
``scripts/run_distributed_demo.py`` meaningful: the hub itself does
nothing to stop an impersonation attempt from reaching every other
operator -- rejecting it is entirely ``identity.verify_signed``'s job,
running independently inside each receiving operator's own process,
against that operator's own trusted public-key registry.

Runs as its own OS process (``python3 -m satcollision.hub ...``), listens
on a real TCP port, and shuts down deterministically once it has relayed
``--expected-messages`` inbound messages (or after ``--max-duration``
seconds, as a safety net against a hung client).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .wire import read_message, send_message


class Hub:
    def __init__(self, expected_messages: int | None = None):
        self.expected_messages = expected_messages
        self.received_count = 0
        self.clients: dict[asyncio.StreamWriter, str | None] = {}
        self._done = asyncio.Event()
        # Every message this hub has ever relayed, replayed in order to
        # each newly connecting client before anything else. This makes
        # the hub a full-history relay (closer to a durable message bus
        # than a live-only broadcast), and it's what makes delivery
        # order-independent: without it, a message sent while some
        # legitimate operator hadn't connected yet would reach only
        # whoever happened to already be online, and a peer that joined
        # even a moment later would silently never see it -- no amount of
        # a receiving node staying connected longer afterwards can recover
        # a message it was never sent in the first place. An earlier
        # version of this hub replayed only "ready" rendezvous pings on
        # this theory (every other message "needs to be seen live"), which
        # was broken exactly this way: in scripts/run_distributed_demo.py,
        # an attacker that connects and sends before every legitimate
        # operator has joined could have its forged message broadcast only
        # to whichever of them were already online, so a late-joining peer
        # would never receive it to reject -- not a timing race a longer
        # wait on the receiving end could fix, since the message was never
        # delivered at all. Replaying full history removes the ordering
        # dependency entirely: every client receives exactly one copy of
        # every other client's message, once, regardless of who connected
        # when.
        self._history: list[dict] = []

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients[writer] = None
        peer = writer.get_extra_info("peername")
        print(f"[hub] connection from {peer}", flush=True)
        # Snapshot history *after* registering this client in self.clients
        # but *before* awaiting anything, so the two delivery paths can't
        # overlap: anything already in the snapshot is replayed exactly
        # once here, and anything appended from this point on is delivered
        # exactly once via the live _broadcast path below (which now sees
        # this client, since it was just registered above). Without the
        # snapshot, a message appended mid-replay would be visible to both
        # this loop (plain list iteration picks up concurrent appends) and
        # to a concurrent _broadcast call for that same message -- a
        # duplicate delivery, not a gap, but still worth ruling out.
        history_snapshot = list(self._history)
        for past_message in history_snapshot:
            await send_message(writer, past_message)
        try:
            while True:
                message = await read_message(reader)
                if message is None:
                    break
                if message.get("type") == "ready":
                    self.clients[writer] = message.get("operator")
                self._history.append(message)
                self.received_count += 1
                print(f"[hub] relaying message {self.received_count} "
                      f"(type={message.get('type')}, claims operator="
                      f"{message.get('sender_operator') or message.get('operator')})", flush=True)
                await self._broadcast(exclude=writer, message=message)
                if self.expected_messages is not None and self.received_count >= self.expected_messages:
                    self._done.set()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            del self.clients[writer]
            writer.close()

    async def _broadcast(self, exclude: asyncio.StreamWriter, message: dict) -> None:
        for client_writer in list(self.clients.keys()):
            if client_writer is exclude:
                continue
            try:
                await send_message(client_writer, message)
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def run(self, host: str, port: int, max_duration: float) -> None:
        server = await asyncio.start_server(self.handle_client, host, port)
        actual_port = server.sockets[0].getsockname()[1]
        print(f"[hub] listening on {host}:{actual_port}", flush=True)
        try:
            await asyncio.wait_for(self._done.wait(), timeout=max_duration)
            print(f"[hub] relayed {self.received_count} expected messages -- shutting down", flush=True)
        except asyncio.TimeoutError:
            print(f"[hub] max_duration={max_duration}s reached with {self.received_count} messages "
                  f"relayed -- shutting down (safety net, not the happy path)", flush=True)
        server.close()
        await server.wait_closed()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-messages", type=int, default=None,
                         help="shut down once this many inbound messages have been relayed")
    parser.add_argument("--max-duration", type=float, default=30.0,
                         help="safety-net shutdown after this many seconds regardless")
    args = parser.parse_args(argv)

    hub = Hub(expected_messages=args.expected_messages)
    asyncio.run(hub.run(args.host, args.port, args.max_duration))
    return 0


if __name__ == "__main__":
    sys.exit(main())
