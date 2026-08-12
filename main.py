from src.cli import run

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
    except Exception as e:
        print(f"\n Error: {e}")