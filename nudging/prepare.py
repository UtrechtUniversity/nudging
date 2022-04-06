""" Determine nudge success per subject using propensity score matching
"""
# pylint: disable=unused-import
import pandas as pd

from nudging.dataset import Balaban, Hotard, Pennycook1, Pennycook2,\
    Lieberoth, Vandenbroele, Milkman # noqa
from nudging.utils import clean_dirs


def combine(infiles, outfile):
    """Combine csv files
    Args:
        infiles (list): list of inputfiles
        outfile (str): name of outputfile
    """
    dataframes = []
    print("\nCombining files:")
    for fin in infiles:
        print(fin)
        dataframes.append(pd.read_csv(fin, encoding="iso-8859-1"))
    data = pd.concat(dataframes)

    # Replace missing values with Nans
    data.replace("", pd.NA, inplace=True)

    data.to_csv(outfile, index=False)

    print(f"Combined dataset in {outfile}")


if __name__ == "__main__":

    # Megastudy interventions, Milkman et al 2021 (TODO: order)
    interventions = [
    'Exercise Commitment Contract Encouraged',
    'Free Audiobook Provided',
    'Free Audiobook Provided, Temptation Bundling Explained',
    'Higher Incentives a',
    # 'Placebo Control',
    'Planning, Reminders & Micro-Incentives to Exercise',
    'Exercise Social Norms Shared (Low but Increasing)',
    'Rigidity Rewarded a',
    'Following Workout Plan Encouraged',
    'Fitness Questionnaire with Decision Support & Cognitive Reappraisal Prompt',
    'Effective Workouts Encouraged', 'Bonus for Consistent Exercise Schedule',
    'Reflecting on Workouts Encouraged',
    'Planning Workouts Encouraged',
    'Defaulted into 3 Weekly Workouts',
    'Exercise Social Norms Shared (High and Increasing)',
    'Asked Questions about Workouts',
    'Exercise Encouraged',
    'Rigidity Rewarded d',
    'Values Affirmation',
    'Rigidity Rewarded c',
    'Exercise Commitment Contract Explained',
    'Exercise Social Norms Shared (Low)',
    'Bonus for Variable Exercise Schedule',
    'Fitness Questionnaire',
    'Exercise Encouraged with Typed Pledge',
    'Mon-Fri Consistency Rewarded, Sat-Sun Consistency Rewarded',
    'Defaulted into 1 Weekly Workout',
    'Exercise Encouraged with Signed Pledge',
    'Planning Workouts Rewarded',
    'Rigidity Rewarded b',
    'Planning Benefits Explained',
    'Choice of Gain- or Loss-Framed Micro-Incentives',
    'Reflecting on Workouts Rewarded',
    'Exercise Fun Facts Shared',
    'Bonus for Returning after Missed Workouts a',
    'Planning Fallacy Described and Planning Revision Encouraged',
    'Exercise Encouraged with E-Signed Pledge',
    'Exercise Social Norms Shared (High)',
    'Planning Revision Encouraged',
    'Loss-Framed Micro-Incentives',
    'Higher Incentives b',
    'Exercise Advice Solicited',
    'Fitness Questionnaire with Decision Support',
    'Exercise Commitment Contract Explained Post-Intervention',
    'Bonus for Returning after Missed Workouts b',
    'Rewarded for Responding to Questions about Workouts',
    'Rigidity Rewarded e',
    'Gain-Framed Micro-Incentives',
    'Fitness Questionnaire with Cognitive Reappraisal Prompt',
    'Gym Routine Encouraged',
    'Fun Workouts Encouraged',
    'Values Affirmation Followed by Diagnosis as Gritty',
    'Exercise Advice Solicited, Shared with Others']

    datasets = {
        # "Simulated": (Simulated, "data/external/simulated/simulated.csv"),
        # "Vandenbroele": (Vandenbroele, "data/external/016_vandenbroele/S2_OpenAccess.sav"),
        # "Pennycook1": (Pennycook1, "data/external/002_pennycook/Pennycook et al._Study 1.csv"),
        # "Pennycook2": (Pennycook2, "data/external/002_pennycook/Pennycook et al._Study 2.csv"),
        # "Hotard": (Hotard, "data/external/004_hotard/NNYFeeWaiverReplicationData.dta"),
        # "Balaban": (Balaban, "data/external/008_balaban/anon1.dta"),
        # "Lieberoth": (Lieberoth, "data/external/011_lieberoth/Commuter experiment_simple.csv"),
        "Milkman": (Milkman, "data/external/021_milkman/Data/StepUp Data/pptdata.csv"),
    }
    
    # Cleanup old data
    outdirs = ["data/raw", "data/interim"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Read and convert each dataset
    for name, dataset_info in datasets.items():
        print(f"\ndataset {name}")
        data_class, data_fp = dataset_info

        if name == "Milkman":
            for nudge_type, intervention in enumerate(interventions):
                dataset = data_class.from_file(data_fp, nudge_type, intervention)
                # Write raw data to csv
                dataset.write_raw("data/raw/" + name + str(nudge_type) + ".csv")
                # Write interim data to csv
                dataset.write_interim("data/interim/" + name + str(nudge_type) + ".csv")

        else:
            dataset = data_class.from_file(data_fp)
            # Write raw data to csv
            dataset.write_raw("data/raw/" + name + ".csv")

            # Write interim data to csv
            dataset.write_interim("data/interim/" + name + ".csv")


