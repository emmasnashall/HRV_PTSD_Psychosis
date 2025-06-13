import pandas as pd
import neurokit2 as nk
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import copy



   
def get_participants(city):
    participants_path = os.path.join("Data", city)
    participants = os.listdir(participants_path)
    return participants

def get_taskfiles(participant, city, timepoint):
    tasks_path = os.path.join("Data", city, participant, timepoint, "Task")
    # if city == "London":
        
    # elif timepoint == "T1":
    #     tasks_path = os.path.join("Data", city, participant, timepoint,  "Task", "renamed")
    # else:
    #     tasks_path = os.path.join("Data", city, participant, timepoint,  "Task")
        
    if os.path.exists(tasks_path) and os.path.isdir(tasks_path):
        task_files = os.listdir(tasks_path)
        #drop taskfiles if they are folders
        task_files = [file for file in task_files if not os.path.isdir(os.path.join(tasks_path, file))]
        task_files = pd.Series(task_files)
    else: 
        task_files = pd.Series([])
    
    return task_files

def get_datafiles(participant, city, timepoint):
    if city == "London":
        data_path = os.path.join("Data", city, participant, timepoint, "physio", "raw")
    else:
        data_path = os.path.join("Data", city, participant, timepoint, "physio", "reidentified_8_seconds")
    
    if os.path.exists(data_path) and os.path.isdir(data_path):
        data_files = os.listdir(data_path)
        data_files = pd.Series(data_files)
    else: 
        data_files = pd.Series([])
    
    return data_files

#load and format data

def format_physio(datafile, city):
    if city == "London":
        data = pd.read_csv(datafile, sep = "	", header = 0, skiprows=[1,2,3,4,5])
    else:
        data = pd.read_csv(datafile)
        data = pd.read_csv(datafile)
        data.rename(columns={"ppu": "PG"}, inplace=True)

    return data
    

# def load_raw(file):
#     data=pd.read_csv(file, sep = "	", header = 0, skiprows=[1,2,3,4,5])
#     ppg_signal = data["PG"] 
#     return data, ppg_signal

# def london_data_format(datafile):
#     data, PG= load_raw(datafile, cutstart = True)
#     data = data[399:]
#     data['Time'] *= 1000
#     data["TimeAtStartOfTrial"]= data["Time"]-8000 # normalise time for merge
#     data["TimeAtStartOfTrial"] = data["TimeAtStartOfTrial"].round().astype(int)
#     return data

# def northern_data_format(datafile):
#     ##write to format data
#     data = pd.read_csv(datafile)
#     data.rename(columns={"ppu": "PG"}, inplace=True)
#     data['Time'] *= 1000
#     data["TimeAtStartOfTrial"] = data["Time"].round().astype(int)
#     return data

#load and format task

#taskfile = r"Data\Newcastle\30434\T1\task\StarD5v2_TrMemory_30434_1_Rpt1.csv"
def format_taskfile(taskfile):
    task = pd.read_csv(taskfile)

    Playlist = task.loc[task["SubjectID"]=="Playlist", "Visit"].values[0]

    ratings = {}


    if "run" in Playlist:
        Avoidance = task.loc[task["RatingText"].str.contains("I avoided thinking about some of my trauma reminders in that session", na=False), "Rating"].values[0] if task["RatingText"].str.contains("I avoided thinking about some of my trauma reminders in that session", na=False).any() else np.nan
        Numbness = task.loc[task["RatingText"] == "During that session I felt spaced out; numb; or emotionally shut down", "Rating"].values[0] if task["RatingText"].str.contains("During that session I felt spaced out; numb; or emotionally shut down", na=False).any() else np.nan
        Anxiety = task.loc[task["RatingText"] == "Right now I feel: Anxious", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Anxious", na=False).any() else np.nan
        
        ratings[f"Avoidance"] = Avoidance
        ratings[f"Numbness"] = Numbness
        ratings[f"Anxiety"] = Anxiety

        task = task.dropna(subset=['TimeAtStartOfTrial'])
        task["TrialDuration"] = task["TimeAtStartOfTrial"].shift(-1)-task["TimeAtStartOfTrial"]
        last_trial_index = max(task.loc[task["TrialType"] == "Trial"].index)
        task = task[:last_trial_index+1]
        task["TimeAtStartOfTrial"] = task["TimeAtStartOfTrial"].round().astype(int)

    if "post" in Playlist:
        Memories = task.loc[task["RatingText"] == "While you were resting; did memories about your trauma come up?", "Rating"].values[0] if task["RatingText"].str.contains("While you were resting; did memories about your trauma come up?", na=False).any() else np.nan
        Vivdness = task.loc[task["RatingText"] == "How vivid were those memories when they came up?", "Rating"].values[0] if task["RatingText"].str.contains("How vivid were those memories when they came up?", na=False).any() else np.nan
        Intrusiveness = task.loc[task["RatingText"] == "How much did those memories come up without you wanting them to?", "Rating"].values[0] if task["RatingText"].str.contains("How much did those memories come up without you wanting them to?", na=False).any() else np.nan
    
        ratings["Memories"] = Memories 
        ratings["Vividness"] = Vivdness
        ratings["Intrusiveness"] = Intrusiveness


    if "pre" in Playlist:
        Cheerful = task.loc[task["RatingText"] == "Right now I feel: Cheerful", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Cheerful", na=False).any() else np.nan
        Anxious = task.loc[task["RatingText"] == "Right now I feel: Anxious", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Anxious", na=False).any() else np.nan
        Sad = task.loc[task["RatingText"] == "Right now I feel: Sad", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Sad", na=False).any() else np.nan
        Satified = task.loc[task["RatingText"] == "Right now I feel: Satisfied", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Satisfied", na=False).any() else np.nan
        Ashamed = task.loc[task["RatingText"] == "Right now I feel: Ashamed", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Ashamed", na=False).any() else np.nan
        Relaxed = task.loc[task["RatingText"] == "Right now I feel: Relaxed", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Relaxed", na=False).any() else np.nan
        Dissociated = task.loc[task["RatingText"] == "Right now I feel: Right now I feel spaced out; numb; or emotionally shut down", "Rating"].values[0] if task["RatingText"].str.contains("Right now I feel: Right now I feel spaced out; numb; or emotionally shut down", na=False).any() else np.nan

        ratings["Cheerful"] = Cheerful
        ratings["Anxious"] = Anxious
        ratings["Sad"] = Sad
        ratings["Satisfied"] = Satified
        ratings["Ashamed"] = Ashamed
        ratings["Relaxed"] = Relaxed
        ratings["Dissociated"] = Dissociated
    
    return task, ratings


#create events from task

def get_trial_info(task, sampling_rate):

    events_onset = ((task["TimeAtStartOfTrial"]/1000)*sampling_rate).tolist() #convert milliseconds to sample indicies at sampling frequency
    events_durations = ((task["TrialDuration"]/1000)*sampling_rate).tolist()
    events_labels = task["TrialNo"].tolist()
    events_conditions = task["IsOwn"].tolist()
    events_nowness = task["Rating"].tolist()

    trial_info = nk.events_create(events_onset, event_labels=events_labels, event_conditions= events_conditions, event_durations= events_durations)
    trial_info["Nowness"] = events_nowness
    
    return trial_info



def ibi_trial_info_format(trial_info, sampling_rate):
    ibi_trial_info = copy.deepcopy(trial_info)
    for i in range(len(ibi_trial_info["onset"])):
        ibi_trial_info["onset"][i] = ibi_trial_info["onset"][i] - (4 * sampling_rate)
        ibi_trial_info["duration"][i] = ibi_trial_info["duration"][i] + (4 * sampling_rate) # reduce each onset by 2s
    for key in ibi_trial_info.keys(): 
        ibi_trial_info[key] = ibi_trial_info[key][1:] # remove first 8 seconds 
    return ibi_trial_info

# get peaks per trial
def trial_peaks(start_time, duration, peak_data): 
    end_time = start_time + duration
    peaks = peak_data
    trial_peaks = peaks[(peaks >= start_time) & (peaks < end_time)]
    return trial_peaks

# divide yes and no peaks UNUSED
# def format_trial_group(events_conditions, condition):
#     No_Yes_trial = [index for index, value in enumerate(events_conditions) if value == condition]
#     No_Yes_trial = [No_Yes_trial[i] +1 for i in range(len(No_Yes_trial))]
#     return No_Yes_trial

#format accelatation by condition
def acceleration_by_trial(acceleration, Yes_trial, No_trial):

    max_length = max(len(acceleration[trial]) for trial in acceleration)

    # Extend each trial with NaNs to match the maximum length
    for trial in acceleration:
        length = len(acceleration[trial])
        difference = max_length - length  # Now max_length is a number
        if difference > 0:
            acceleration[trial] = np.append(acceleration[trial], [np.nan] * difference)

    dataframe = pd.DataFrame(acceleration)
    #dataframe = dataframe.drop(columns=0)

    Yes_data = dataframe.drop(columns=[str(trial) for trial in No_trial if trial in dataframe.columns], errors='ignore')
    No_data = dataframe.drop(columns=[str(trial) for trial in Yes_trial if trial in dataframe.columns], errors='ignore')
    return Yes_data, No_data

# IBI df calculations
def mean_sem_samplepresent(condition):
    condition["mean"] = condition.mean(axis = 1)
    condition['SEM'] = condition.std(axis=1, skipna=True) / np.sqrt(condition.count(axis=1))
    condition["sample_present"] = condition.notna().sum(axis=1) / condition.shape[1] * 100
    condition["Beat_Number"] = range(-1, len(condition) - 1)
    # condition["Beat_Number"] = condition["Beat_Number"].astype(str)
    # condition["Beat_Number"][0] = "-3"
    return condition

#plot signal

def plot_ppg(epochs, folderpath, sampling_rate):
    for epoch in epochs: 
        df = epochs[epoch]  # Get the DataFrame for the current epoch
        if df.empty: 
            continue
        else:
            df.reset_index(drop=True, inplace=True)
            ppg_peaks = df["PPG_Peaks"]  # Extract PPG Peaks from the DataFrame
            peak_row = np.where(ppg_peaks == 1)[0]
            condition = df["Condition"].iloc[0]
            if condition == "Yes":
                filename = f'{epoch}.Trauma.html'
            else:
                filename = f'{epoch}.Control.html'
            # Create an info dictionary with peaks info
            info = {"PPG_Peaks": peak_row,
                    "sampling_rate": sampling_rate}  # Indices of peaks where condition is met
            plot = nk.ppg_plot(df, info, static=False) 
            filepath = os.path.join(folderpath, filename)
            pyo.plot(plot, filename=filepath, auto_open=False)
        
def IBI_plot(Yes_data, No_data, savelocation):
        
    Yes_filtered = Yes_data[Yes_data["sample_present"] == 100]
    Yes_len = len(Yes_filtered)
    No_filtered = No_data[No_data["sample_present"] == 100]
    No_len = len(No_filtered)
    
    lengths = [Yes_len, No_len]
    if Yes_len > No_len:
        x_ticks = Yes_filtered["Beat_Number"]
    else:      
        x_ticks = No_filtered["Beat_Number"]

    Yes_mean = np.array(Yes_filtered["mean"])
    No_mean = np.array(No_filtered["mean"])
    means = [Yes_mean, No_mean]

    Yes_sem = np.array(Yes_filtered["SEM"])
    No_sem = np.array(No_filtered["SEM"])
    SEMs = [Yes_sem, No_sem]


    labels = ["Own", "Control"]

    # Create line plot with SEM
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 5))
    for mean, SEM, length, label in zip(means, SEMs, lengths, labels):
        x = np.arange(-1, length-1)
        ax.plot(x, mean, label= label , marker='o', )
        ax.fill_between(x, mean - SEM, mean + SEM, alpha=0.2, label='SEM')
    
    ax.legend()
    plt.xlabel("Heart Beat Number")
    plt.ylabel("Proportion of 1st Inter-beat Interval")
    plt.xticks(x_ticks)
    plt.savefig(savelocation)
    plt.close(fig)

def combine_epochs(epochs):
    combined = pd.DataFrame()  # Initialize an empty DataFrame

    for dictkey, df in epochs.items():
        if not isinstance(df, pd.DataFrame):
            print(f"Warning: {dictkey} is not a DataFrame and will be skipped.")
            continue  # Skip non-DataFrame items

        # Add a new column to the DataFrame with the key
        df['Trial'] = dictkey  # Add the key as a new column

        # Concatenate the modified DataFrame with the combined DataFrame
        combined = pd.concat([combined, df], ignore_index=True)

    return combined

def load_raw(file, cutstart):
    if cutstart: 
        filename =  file
        data=pd.read_csv(filename, sep = "	", header = 0, skiprows=[1,2,3,4,5])
        ppg_signal = data["PG"] 
        ppg_signal = ppg_signal[399:] # 8s onwards
        return data, ppg_signal
    else: 
        filename =  file
        data=pd.read_csv(filename, sep = "	", header = 0, skiprows=[1,2,3,4,5])
        ppg_signal = data["PG"] 
        return data, ppg_signal
    
def inclusion_checker(label, quality_column):
                
                average = quality_column.mean()

                new_row = {
                    "Label": label,
                    "Avg_Quality": average,
                    "Include": average >= 0.86
                }
                return new_row

def beat_quality(ppg_processed, sample_rate):
    segment = nk.ppg_segment(ppg_processed["PPG_Clean"], peaks=None, sampling_rate=sample_rate)
    segment_quality = {}

    for key, item in segment.items():
        merged_item = pd.merge(item, ppg_processed, how = "left", right_index = True, left_on = "Index")
        merged_item["Time"] = merged_item["Index"]/sample_rate
        segment_quality[key] = merged_item
        
    return segment_quality


def beat_graph(segment_quality_individual, sample_rate, savepathname):
    numbers = list(range(len(segment_quality_individual)))
    time =  [num / sample_rate for num in numbers]

    quality_avg = round(np.mean(segment_quality_individual["PPG_Quality"]), 5)
    
    peak_index = segment_quality_individual.loc[segment_quality_individual['PPG_Peaks'] == 1].index[0]
    peak_row = segment_quality_individual.index.get_loc(peak_index)
    peak_int = int(peak_row)
    peak_time = time[peak_int]

    np.where(segment_quality_individual["PPG_Peaks"]==1)

    # Create the line graph
    plt.figure(figsize=(8, 5))
    plt.plot(time, segment_quality_individual["Signal"], marker="o", linestyle="-", color="blue", label="PPG_clean")
    plt.plot(time, segment_quality_individual["PPG_Raw"], marker="o", linestyle="-", color="red", label="PPG_signal")
    
    # Draw a vertical line at peak
    plt.axvline(x=peak_time, color='green', linestyle='--', label='Peak')

    # Adding labels and title
    plt.xlabel("Time")
    plt.ylabel("PPG Signal")
    plt.title(f"PPG Raw and Cleaned, Quality Average {quality_avg}")
    plt.legend()
    plt.grid(True)

    plt.savefig(savepathname)
    #plt.show()
    plt.close()
    # Show the graph
    

# data = pd.read_csv("Processed/Neurokit2/test/London/P010095/T1/run1/ppg_processed_P010095_run1_T1.csv")

# beat_quality_output = beat_quality(data, 50)
# beat_graph(beat_quality_output["1"], 50, "test.png")


def inclusion_criteria(row, TD=True, quality_threshold=0.86):
    Min_num_peaks = True if row["Min_num_peaks"] == True else False
    BPM_in_range = True if row["BPM_in_range"] == True else False
    Proportion_Below_Quality = True if row["Proportion_Below_Quality"] < 0.5 else False
    sus_missed_beat = True if row["sus_missed_beat"] == False else True
    average_quality = True if row["average_quality"] > quality_threshold and row["average_quality"] < 1 else False
    RMSSD_in_range = True if row.get("RMSSD_in_range", True) == True else False

    if Min_num_peaks and BPM_in_range  and sus_missed_beat and (RMSSD_in_range if TD else True):
        if not average_quality or not Proportion_Below_Quality:
            row["Include"] = "REVIEW"
        else:
            row["Include"] = True
    else:
        row["Include"] = False

    return row["Include"]


def get_files(participant, city, timepoint):
    with ThreadPoolExecutor() as executor:
        taskfiles_future = executor.submit(get_taskfiles, participant, city, timepoint)
        datafiles_future = executor.submit(get_datafiles, participant, city, timepoint)
        taskfiles = taskfiles_future.result()
        datafiles = datafiles_future.result()
    return taskfiles, datafiles

def add_to_log(log, hrv_summary_df):

    inlcuded_whole_segments = int(((hrv_summary_df["Include"] == True) & (hrv_summary_df["subset_used"] == "Not needed")).sum())
    inlcuded_partial_segments = int(((hrv_summary_df["Include"] == True) & (hrv_summary_df["subset_used"] == True)).sum())
    Excluded_Trials = int(len(hrv_summary_df) - inlcuded_whole_segments - inlcuded_partial_segments)
    For_Review_Count = int((hrv_summary_df["Include"] == "REVIEW").sum())
    
    log["Included_Trials"] = log.get("Included_Trials",{})
    log["Included_Trials"]["Fully_processed"] = log["Included_Trials"].get("Fully_processed", inlcuded_whole_segments)
    log["Included_Trials"]["subset_used"] = inlcuded_partial_segments
    log["Excluded"] = log.get("Excluded", Excluded_Trials)
    log["For_Review"] = log.get("For_Review", For_Review_Count)

    return log



def add_quality_columns(hrv_row_df, segment_df, below_threshold_index, TD = True, sampling_rate=None):
    max_HR = 190
    min_HR = 30 
    max_RMSSD = 262
    min_RMSSD = 5
    
    
    peak_indices = segment_df.loc[segment_df["PPG_Peaks"] == 1, "Index"].values
    
    hrv_row_df["average_quality"] = segment_df["PPG_Quality"].mean()
    if TD:
        hrv_row_df["RMSSD_in_range"] = (hrv_row_df["HRV_RMSSD"].iloc[0]>min_RMSSD) and (hrv_row_df["HRV_RMSSD"].iloc[0] < max_RMSSD) if "HRV_RMSSD" in hrv_row_df else "NA"
    
    # hrv_row_df["Below_Quality_Beats"] = pd.Series(peak_indices).isin(below_threshold_index).sum()
    hrv_row_df["Number_of_Beats"] = len(peak_indices)
    hrv_row_df["Proportion_Below_Quality"] = hrv_row_df["Below_Quality_Beats"]/hrv_row_df["Number_of_Beats"]
    hrv_row_df["Min_num_peaks"] =  len(peak_indices)> 2 and  len(peak_indices) > ((len(segment_df)/sampling_rate/5) +2) # criteria from Rapid HRV toolbox
    hrv_row_df["BPM_in_range"] =  (max(segment_df["PPG_Rate"])<max_HR) and (min(segment_df["PPG_Rate"])>min_HR)

    if len(peak_indices) == 0:
        hrv_row_df["sus_missed_beat"] = "NA"
    else:
        hrv_row_df["sus_missed_beat"] = False
        for i in range(len(peak_indices)-1):
            peak_distance = (peak_indices[i+1] - peak_indices[i])/sampling_rate
            if peak_distance > 3:
                hrv_row_df["sus_missed_beat"] = True
                break
                
    return hrv_row_df

if __name__ == "__main__":
    # Example usage
    city = "London"
    participant = "P010095"
    timepoint = "T1"
    taskfiles, datafiles = get_files(participant, city, timepoint)
    print("Task files:", taskfiles)
    print("Data files:", datafiles)