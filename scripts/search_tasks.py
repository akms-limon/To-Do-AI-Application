from app.database import initialize_database
from app.search_service import search_tasks


def main():
    initialize_database()
    query = input("Search tasks: ")

    matches = search_tasks(query)

    if not matches:
        print("No relevant tasks found.")
        return

    print("\nRelevant candidates (up to 5):")
    for match in matches:
        task = match["task"]

        print(
            f"#{task['id']} | {task['title']} | "
            f"{task['status']} | distance={match['distance']:.3f}"
        )


if __name__ == "__main__":
    main()
