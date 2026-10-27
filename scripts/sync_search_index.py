from app.database import initialize_database
from app.search_service import synchronize_index


def main():
    initialize_database()
    result = synchronize_index()

    print("Search index synchronized:")
    for name, count in result.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()