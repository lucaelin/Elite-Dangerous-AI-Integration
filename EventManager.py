import dataclasses
import sys
import json
from typing import Dict, List, Literal, Callable

from EDJournal import *
from Event import GameEvent, Event, ConversationEvent, ToolEvent
from Logger import log


class EventManager:
    def __init__(self, on_reply_request: Callable[[List[Event], List[Event]], any], game_events: List[str], continue_conversation: bool = False):
        self.pending: List[Event] = []
        self.processed: List[Event] = []
        self.is_replying = False
        self.on_reply_request = on_reply_request
        self.game_events = game_events

        if continue_conversation:
            self.load_history()
        else:
            log('Debug', 'Starting a new conversation.')

    def add_game_event(self, content: Dict):
        event = GameEvent(content=content)
        self.pending.append(event)
        log('Event', event)
        return self._handle_new_event()

    def add_conversation_event(self, role: Literal['user', 'assistant'], content: str):
        event = ConversationEvent(kind=role, content=content)
        self.pending.append(event)
        if role == 'user':
            log('CMDR', content)
        elif role == 'assistant':
            log('COVAS', content)
        return self._handle_new_event()

    def add_assistant_complete_event(self):
        event = ConversationEvent(kind='assistant_completed', content='')
        self.pending.append(event)
        self.is_replying = False
        # log('Debug', event)
        return self._handle_new_event()

    def add_tool_call(self, request: Dict, results: List[Dict]):
        event = ToolEvent(request=request, results=results)
        self.pending.append(event)
        log('Action', [result['name']+': ' + result['content'] for result in results])
        return self._handle_new_event()

    def _handle_new_event(self):
        self.save_history()
        if not self.is_replying and self.should_reply():
            self.is_replying = True
            new_events = self.pending
            self.processed += self.pending
            self.pending = []
            # log('Debug', 'eventmanager requesting reply')
            sys.stdout.flush()
            self.on_reply_request(self.processed, new_events)
            return True

        return False

    def should_reply(self):
        if len(self.pending) == 0:
            return False

        for event in self.pending:
            # check if pending contains conversational events
            if isinstance(event, ConversationEvent) and event.kind == "user":
                return True

            if isinstance(event, ToolEvent):
                return True

            if isinstance(event, GameEvent) and event.content.get("event") in self.game_events:
                return True

        return False

    def save_history(self):
        class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    return super().default(o)

        with open('history.json', 'w') as json_file:
            json.dump(self.processed[-1000:] + self.pending, json_file, cls=EnhancedJSONEncoder)

    def load_history(self):
        try:
            with open('history.json', 'r') as json_file:
                history = json.load(json_file)
                log('Debug', f'Continuing conversation with {len(history)} elements.')
        except json.JSONDecodeError:
            log('error', 'Error while loading history.json')
            history = []
        except FileNotFoundError:
            history = []

        for rawevent in history:
            if rawevent["kind"] in ['user', 'assistant', 'assistant_completed']:
                self.processed.append(ConversationEvent(**rawevent))
            if rawevent["kind"] == 'tool':
                self.processed.append(ToolEvent(**rawevent))
            if rawevent["kind"] == 'game':
                self.processed.append(GameEvent(**rawevent))
