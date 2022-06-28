import argparse
from termcolor import colored


arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "blue_score",
        "meteor_score", "count_parameters", "ter_score", "confusion_matrix"
    ],
    help="Add an action to run this project"
)

arg_pr.add_argument(
    "-s", "--source", required=True,
    choices=[
        "en", "cv"
    ],
    help="Source languague for the translation"
)

arg_pr.add_argument(
    "-t", "--target", required=True,
    choices=[
        "en", "cv"
    ],
    help="Target languague for the translation"
)

args = vars(arg_pr.parse_args())


if args["source"] == args["target"]:
    print(
        colored("Error: Source languague and Target languague should not be the same.", "red", attrs=["bold"])
    )
    exit(1)


from src.gru import Seq2Seq_Translator
from src.utils import check_dataset

check_dataset()
gru_attention_translator = Seq2Seq_Translator(args["source"], args["target"])


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": gru_attention_translator.console_model_test,
        "train": gru_attention_translator.train_model,
        "test_model": gru_attention_translator.test_model,
        "blue_score": gru_attention_translator.calculate_blue_score,
        "meteor_score": gru_attention_translator.calculate_meteor_score, 
        "count_parameters": gru_attention_translator.count_hyperparameters,
        "ter_score": gru_attention_translator.calculate_ter,
        "confusion_matrix": gru_attention_translator.generate_confusion_matrix
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
