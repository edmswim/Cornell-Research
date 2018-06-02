import argparse
from run import run_classify
from run import run_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simpleNN')
    parser.add_argument('--participantDependent', type=str, default='True')
    parser.add_argument('--totalDays', type=int, default=3)
    parser.add_argument('--task', type=str, default="classify")
    parser.add_argument('--normalize', type=str, default="z-score")
    parser.add_argument('--leave_one_patient', type=str, default='False')
    args = parser.parse_args()

    if args.task == "classify":
        run_classify.run(args.participantDependent, args.leave_one_patient, args.totalDays, args.normalize, args.model)

    elif args.task == "regression":
        run_regression.run(args.participantDependent, args.leave_one_patient, args.totalDays, args.normalize, args.model)
