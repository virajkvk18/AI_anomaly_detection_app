import subprocess

while True:

    print("\nAI Anomaly Detection Platform")
    print("1 Generate dataset")
    print("2 Train model")
    print("3 Start API")
    print("4 Run tester")
    print("5 Exit")

    choice=input("Select option:")

    if choice=="1":
        subprocess.run(["python","data/collect_metrics.py"])

    elif choice=="2":
        subprocess.run(["python","model/train_model.py"])

    elif choice=="3":
        subprocess.run(["uvicorn","api.app:app","--reload"])

    elif choice=="4":
        subprocess.run(["python","tester/tester.py"])

    elif choice=="5":
        break