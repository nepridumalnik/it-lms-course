from app.controller import AppController

if __name__ == "__main__":
    try:
        app = AppController()
        app.run()
    except Exception as e:
        print(f"Failed with exception {e}")
        exit(-1)
