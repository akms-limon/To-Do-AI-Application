"""Interactive check using the same receipt/memory path as the website."""

from uuid import UUID, uuid4
from app.chat_service import ChatRequest, ChatServiceError, handle_chat
from app.database import initialize_database


def main():
    initialize_database()
    print("TaskFlow: this script executes changes in your task database.")
    raw = input("Conversation UUID [new]: ").strip()
    thread = UUID(raw) if raw else uuid4()
    print("Conversation:", thread)
    while True:
        text = input("You (exit to leave): ").strip()
        if text.lower() == "exit":
            break
        if not 1 <= len(text) <= 2000:
            print("Enter 1–2000 characters.")
            continue
        payload = ChatRequest(thread_id=thread, message_id=uuid4(), message=text)
        while True:
            try:
                print("Assistant:", handle_chat(payload)["reply"])
                break
            except ChatServiceError as error:
                print(error.detail)
                if input("Retry same request? [yes/no]: ").strip().lower() != "yes":
                    print("Keep this request ID for recovery:", payload.message_id)
                    return


if __name__ == "__main__":
    main()
