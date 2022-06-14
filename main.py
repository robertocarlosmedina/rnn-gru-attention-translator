import argparse

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "flask_api", "blue_score",
        "meteor_score", "count_parameters", "ter_score", "confusion_matrix"
    ],
    help="Add an action to run this project"
)
args = vars(arg_pr.parse_args())


from src.translator import Seq2Seq_Translator
from src.flask_api import Resfull_API


gru_attention_translator = Seq2Seq_Translator()


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": gru_attention_translator.console_model_test,
        "train": gru_attention_translator.train_model,
        "test_model": gru_attention_translator.test_model,
        "flask_api": Resfull_API.start,
        "blue_score": gru_attention_translator.calculate_blue_score,
        "meteor_score": gru_attention_translator.calculate_meteor_score, 
        "count_parameters": gru_attention_translator.count_hyperparameters,
        "ter_score": gru_attention_translator.calculate_ter,
        "confusion_matrix": gru_attention_translator.generate_confusion_matrix
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
