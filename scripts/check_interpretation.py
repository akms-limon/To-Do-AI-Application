from app.llm import interpret_message


def main():
    message = input("Task instruction: ")

    proposal = interpret_message(message)

    print(proposal.model_dump_json(indent=2))
    print("\nPreview only: no task data was changed.")


if __name__ == "__main__":
    main()