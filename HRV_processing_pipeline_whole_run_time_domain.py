

import os
import pandas as pd
import numpy as np
from Scripts.neurokit.functions import get_datafiles, get_taskfiles, load_raw, beat_quality, add_quality_columns, inclusion_criteria, format_physio
import neurokit2 as nk
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

timepoints = ["T1","T2"]
locations = [ "London", "Manchester", "Newcastle"]  # London already processed
quality_threshold = 0.86

for city in locations:
    runs = ["Run1", "pre task", "post task", "Run2", "Run3"] if city == "London" else ["Run1", "Run2", "Run3", "Rest_Pre", "Rest_Post"]
    participants = os.listdir(os.path.join("Data", city))
    for participant in participants:
        for timepoint in timepoints:
            
            #print(f"Processing {city}, {participant}, {timepoint}")

            datafiles = get_datafiles(participant, city, timepoint)
            
            if datafiles.empty:
                continue

            savefolder = os.path.join("Processed", "Neurokit2", "final", city, participant, timepoint)
            os.makedirs(savefolder, exist_ok=True)

            HRV_time = pd.DataFrame()

            for run in runs:
                datafile_bool = datafiles.str.contains(run, case=False)
                datafile_index = np.where(datafile_bool)[0]

                if len(datafile_index) == 1:
                    run_datafile = datafiles.iloc[datafile_index[0]]

                    if city == "London":
                        data_path = os.path.join("Data", city, participant, timepoint, "physio", "raw", run_datafile)
                        sampling_rate = 50
                    else:
                        data_path = os.path.join("Data", city, participant, timepoint, "physio", "reidentified_8_seconds", run_datafile)
                        sampling_rate = 496

                    data = format_physio(data_path, city)
                else:
                    continue
                
                #CHECK IF FILE IS ALREADY PROCESSED
                TD_path = os.path.join(savefolder, "whole_run", f"Whole_run_{participant}_{timepoint}_Time_Domain.csv")
                if os.path.exists(TD_path):
                    check = pd.read_csv(TD_path)
                    if len(check) > 0:
                        print(f"File already processed: {TD_path}")
                        continue
                    else:
                        os.remove(TD_path)
                    

                preprocess_all, ppg_info = nk.ppg_process(data["PG"], sampling_rate=sampling_rate)
                preprocess = preprocess_all[8 * sampling_rate:]  # Remove first 8 seconds

                quality = beat_quality(preprocess, sampling_rate)

                # Time-Domain HRV Processing
                average_quality_run = np.mean(preprocess["PPG_Quality"])
                subset_start, subset_end, subset_duration = np.nan, np.nan, np.nan
                longest_duration, longest_start = 0, 0

                if average_quality_run < quality_threshold:
                    min_duration = sampling_rate * 60  # 1 minute
                    quality_array = np.array(preprocess["PPG_Quality"])

                    iteration_durations = list(range(min_duration, len(quality_array), sampling_rate))
                    iteration_starts = list(range(0, len(quality_array) - min_duration, sampling_rate))

                    for duration in iteration_durations:
                        for start in iteration_starts:
                            if start + duration > len(quality_array):
                                continue

                            segment = quality_array[start: start + duration]
                            segment_average = np.mean(segment)

                            if segment_average > quality_threshold:
                                longest_duration = int(duration)
                                longest_start = start
                                break

                    if longest_duration > 0:
                        quality_segment = preprocess[
                            (preprocess.index >= longest_start) &
                            (preprocess.index <= (longest_start + longest_duration))
                        ]
                        subset_start = longest_start / sampling_rate
                        subset_end = (longest_start + longest_duration) / sampling_rate
                        subset_duration = longest_duration / sampling_rate
                    else:
                        quality_segment = preprocess
                else:
                    quality_segment = preprocess

                peaks = np.where(quality_segment["PPG_Peaks"])[0]
                quality_segment["Index"] = quality_segment.index
                avg_quality_peaks = []
                peak_indices = np.where(quality_segment["PPG_Peaks"] == 1)[0]
                
                for i in range(len(peak_indices) - 1):
                    start = peak_indices[i]
                    end = peak_indices[i + 1]
                    avg_quality = np.mean(quality_segment["PPG_Quality"].iloc[start:end])
                    avg_quality_peaks.append(avg_quality)

                #get indices of peaks below threshold
                below_threshold_peaks = np.where(np.array(avg_quality_peaks) < quality_threshold)[0]
                
                below_threshold_index = peak_indices[below_threshold_peaks]
                if len(peaks) > 2:
                    HRV_time_each = nk.hrv_time(peaks, sampling_rate=sampling_rate)
                    HRV_time_each["Below_Quality_Beats"] = len(below_threshold_index)
                    HRV_time_each = add_quality_columns(HRV_time_each, quality_segment, below_threshold_index, TD=True, sampling_rate=sampling_rate)
                    HRV_time_each["Include"] = HRV_time_each.apply(inclusion_criteria, axis=1, TD=True)
                    HRV_time_each["Proportion_Below_Quality"] = HRV_time_each["Below_Quality_Beats"][0]/HRV_time_each["Number_of_Beats"][0]
                    HRV_time_each["Include"] = inclusion_criteria(HRV_time_each.iloc[0], TD=True, quality_threshold=quality_threshold)
                    HRV_time_each["Series"] = run
                    HRV_time_each["subset_start"] = subset_start
                    HRV_time_each["subset_end"] = subset_end
                    HRV_time_each["subset_duration"] = subset_duration
                    HRV_time = pd.concat([HRV_time, HRV_time_each], axis=0, ignore_index=True)

            # Save Time-Domain HRV results
            HRV_time.to_csv(TD_path)
            print(f"Saved Time-Domain HRV results to {TD_path}")

        Processed.append(participant)


