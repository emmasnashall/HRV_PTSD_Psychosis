Processed = []

import os
import pandas as pd
import numpy as np
#from functions import get_datafiles, get_taskfiles, london_data_format, northern_data_format, load_raw, beat_quality, beat_graph
from Scripts.neurokit.functions import ibi_trial_info_format, get_datafiles, get_taskfiles, load_raw, beat_quality, beat_graph, plot_ppg, format_taskfile, get_trial_info, acceleration_by_trial, mean_sem_samplepresent, IBI_plot, combine_epochs, add_quality_columns, inclusion_criteria, add_to_log, get_files, format_physio
import json
import neurokit2 as nk
import matplotlib.pyplot as plt
import warnings
import plotly.offline as pyo
import copy


### DONE: check start times across sites for physio files. do these align with fmri and task logically
## DONE: get subjective ratings
## DONE: Update inclusion criteria 
## DONE Update loading of renamed data. check if cutting is still appropriate 
## DONE: NEWCASTLE TASK FILES NOT LOADING

warnings.simplefilter(action='ignore', category=FutureWarning)


# city = "London"
# participant = "20189"
# timepoint = "T1"
# run = "Run1"

timepoints = ["T1", "T2"]
locations = ["Manchester", "Newcastle" ] # London already processed

quality_threshold = 0.86


for city in locations:
    runs = ["Run1", "pre task", "post task", "Run2", "Run3"] if city == "London" else ["Run1", "Run2", "Run3", "Rest_Pre", "Rest_Post"]
    #get participants list
    participants = os.listdir(os.path.join("Data", city))
    for participant in participants:
        if participant in Processed:
            continue
        for timepoint in timepoints:
            if timepoint == "T1":
                continue

            print(f"processing {city}, {participant}, {timepoint}")

            #taskfiles, datafiles = get_files(participant, city, timepoint)
            taskfiles = get_taskfiles(participant, city, timepoint)
            datafiles = get_datafiles(participant, city, timepoint)
            
            #nametaskfiles
            task_dict = {}

            for taskfile in taskfiles:
                task_check = pd.read_csv(os.path.join("Data", city, participant, timepoint, "task", taskfile))
                if "Playlist" not in task_check["SubjectID"].values:
                    continue
            
                Playlist = task_check.loc[task_check["SubjectID"]=="Playlist", "Visit"].values[0]

                task_mapping = {
                    "_resting scan1_pre task": "resting_scan1_pre_task",
                    "TraumaMemory_run1": "Run1",
                    "TraumaMemory_run2": "Run2",
                    "TraumaMemory_run3": "Run3",
                    "_resting_scan2_post_task": "resting_scan2_post_task"
                }

                for key, value in task_mapping.items():
                    if key in Playlist:
                        if value in task_dict:
                            task_dict[value].append(taskfile)
                        else:
                            task_dict[value] = [taskfile]
                        break
            # Log number of files present
            log = {
                "data_presence": {
                    "datafiles": len(datafiles),
                    "taskfiles": len(taskfiles)
                }
            }
            
            if taskfiles.empty:
                print(f"{participant}_taskfiles_empty")
                continue 

            print(f"{participant}_taskfile:{taskfiles}")
            # Create dataframe to store HRV data
            HRV_frequency = pd.DataFrame()

            # Create folders to save results
            savefolder = os.path.join("Processed", "Neurokit2", "final", city, participant, timepoint)
            os.makedirs(savefolder, exist_ok=True)

            child_folders = ["beat_quality", "whole_run", "event_related", "plots"]
            for folder in child_folders:
                new_path = os.path.join(savefolder, folder)
                os.makedirs(new_path, exist_ok=True)
            plot_folders = ["Frequency_domain", "PPG_time", "PPG_event", "IBI"]
            for folder in plot_folders:
                plot_path = os.path.join(savefolder,"plots", folder)
                os.makedirs(plot_path, exist_ok=True)

            #check whether datafiles are present, continue to processing if so, otherwise save log and move on
            if datafiles.empty: # add following for event related loop  - if taskfiles.empty:
                savelog = os.path.join(savefolder, f"Processing_log_{participant}_{timepoint}.json")                         
                with open(savelog, "w") as f:
                    json.dump(log, f, indent=4)
                continue
            
            Ratings_all = {}

            for run in runs: 
                run_taskfile = None

                # if run != "Run1":
                #     continue
                # get the index of file with matching run in title
                

                datafile_bool = datafiles.str.contains(run, case = False)
                datafile_index = np.where(datafile_bool)[0]

                # Check if we found exactly one task and data file
                if  len(datafile_index) == 1:
                    # Extract the single match as a string
                    run_datafile = datafiles.iloc[datafile_index[0]]
                    # Create the paths, load and format data

                    if city == "London": 
                        data_path = os.path.join("Data", city, participant, timepoint, "physio", "raw", run_datafile)
                        sampling_rate = 50
                    else:
                        data_path = os.path.join("Data", city, participant,timepoint, "physio", "reidentified_8_seconds", run_datafile)
                        sampling_rate = 496
                    
                    data = format_physio(data_path, city)
                else:
                    continue

                #preprocess data      
                preprocess_all, ppg_info = nk.ppg_process(data["PG"], sampling_rate = sampling_rate)
                
                preprocess_save = os.path.join(savefolder, "whole_run", f"{participant}_{timepoint}_{run}_preprocessed.csv")
                preprocess_all.to_csv(preprocess_save)

                #Quality check preprocessed data
                preprocess = preprocess_all[8*sampling_rate:] # remove first 8 seconds

                quality = beat_quality(preprocess, sampling_rate)
                
                #Identify which beats are below threshold
                below_threshold_keys = [] # add to json dict
                below_threshold_time = []
                below_threshold_index = []
                quality_df = pd.DataFrame()

                for beat in quality:
                    beat_data = quality[beat]
                    average_quality = np.mean(beat_data["PPG_Quality"])

                    peak_index = beat_data["Index"].iloc[9]
                    peak_time =  beat_data["Time"].iloc[9] 
                    peak_index_continuous = beat_data["Index"].iloc[9]
                    
                    start_time = beat_data["Time"].iloc[0]
                    end_time = beat_data["Time"].iloc[-1]

                    peak_distance = []
                    sus_missed_peak = []

                    if int(beat) > 1:
                        previous_beat = str(int(beat)-1)
                        previous_data = quality[previous_beat]

                        previous_peak_index = np.where(previous_data["PPG_Peaks"]==1)[0]
                        if previous_peak_index.size == 0:
                            previous_peak_time = np.nan
                            peak_distance = np.nan
                            sus_missed_peak = np.nan
                        else:
                            previous_peak_time = previous_data["Time"].iloc[previous_peak_index[0]]

                            peak_distance = peak_time - previous_peak_time
                            sus_missed_peak = peak_distance > 3

                        
                    
                    #create plots for below threshold beats
                    if average_quality < quality_threshold:
                        below_threshold_keys.append(beat)
                        below_threshold_time.append(peak_time)
                        below_threshold_index.append(peak_index)
                        """ irregular beat plots removed as potentially unneccessary 
                        beatplot_savefolder = os.path.join(savefolder, "plots", "quality")
                        os.makedirs(beatplot_savefolder, exist_ok=True)
                        beatplot_savepath = os.path.join(beatplot_savefolder, f"beat_quality_{participant}_{run}_{timepoint}_start_{start_time}_end{end_time}_beat_{beat}.png" )
                        beat_graph(data,sampling_rate, beatplot_savepath)
                        """

                    #add quality analysis to dataframe 
                    beat_quality_dict = {
                        "beat_number": int(beat),
                        "start_time": start_time,
                        "end_time": end_time,
                        "peak_time": peak_time,
                        "peak_index": peak_index_continuous,
                        "average_quality": average_quality,
                        "max_quality": max(beat_data["PPG_Quality"]),
                        "min_quality": min(beat_data["PPG_Quality"]),
                        "peak_distance_to prior": peak_distance,
                        "suspected_missed_peak": sus_missed_peak
                    }
                    quality_df = quality_df.append(beat_quality_dict, ignore_index = True)
                    

                #Process Frequency Domain################################
                ##assess quality of signal 
                average_quality_run = np.mean(preprocess["PPG_Quality"])
                

                subset_start = np.nan
                subset_end = np.nan
                subset_duration = np.nan

                longest_duration = 0
                longest_start = 0

                ### if quality is below threshold, see if there is a section above 1 min that is above threshold 
                if average_quality_run < quality_threshold:
                    min_duration = sampling_rate * 60 # 1 minute
                    quality_array = np.array(preprocess["PPG_Quality"]) # to be iterated through

                    iteration_durations = list(range(min_duration, len(quality_array), sampling_rate))
                    iteration_starts = list(range(0, len(quality_array)-min_duration, sampling_rate))

                    for duration in iteration_durations:
                        #print(f"Processing {duration} duration")
                        for start in iteration_starts:
                            # Check if there's enough data for this segment
                            if start + duration > len(quality_array):
                                #print(f"Not enough data for start={start} and duration={duration}")
                                continue
                            
                            # Compute the segment and its average
                            segment = quality_array[start : start + duration]
                            segment_average = np.mean(segment)
                        
                            # Check if this segment satisfies the condition
                            if segment_average > quality_threshold:
                                longest_duration = int(duration)
                                longest_start = start
                                break
                        
                    #check if a subsection was identified
                    
                    if longest_duration > quality_threshold: #below quality threshold and useable segment
                        ## segment data according to quality checks
                        fq_data = ppg_info["PPG_Peaks"][
                            (ppg_info["PPG_Peaks"]>= longest_start) & 
                            (ppg_info["PPG_Peaks"]<= (longest_start + longest_duration))]
                        
                        quality_segment = preprocess[
                            (preprocess.index>= longest_start) & 
                            (preprocess.index<= (longest_start + longest_duration))]
                        
                        subset_used = True
                        if longest_start > 0:
                            subset_start = longest_start/sampling_rate
                        else:
                            subset_start = 0
                        
                        subset_end = (longest_start + longest_duration)/sampling_rate
                        subset_duration = longest_duration/sampling_rate

                    else: #below quality threshold but no useable segment
                        fq_data = ppg_info["PPG_Peaks"]
                        quality_segment = preprocess
                        subset_used = "None avaliable"
                        
                else: #above quality threshold
                    fq_data = ppg_info["PPG_Peaks"]
                    quality_segment = preprocess
                    subset_used = "Not needed"
                
                ## process HRV
                HRV_frequency_each = nk.hrv_frequency(fq_data, sampling_rate = sampling_rate)
                quality_segment["Index"] = quality_segment.index
                HRV_frequency_each = add_quality_columns(HRV_frequency_each, quality_segment, below_threshold_index, TD = False, sampling_rate = sampling_rate)
                
                # ##save FD plots
                # save_address = os.path.join(savefolder,"plots", "Frequency_Domain")
                # os.makedirs(save_address, exist_ok=True)
                # save_plot_name = os.path.join(save_address, f"{participant}_{timepoint}_hrv_frequency_{run}.png")
                # plt.savefig(save_plot_name, dpi=300) 
                # plt.close()
                
                ## add to FD dataframe
                HRV_frequency_each["subset_used"] = subset_used
                HRV_frequency_each["subset_start"] = subset_start
                HRV_frequency_each["subset_end"] = subset_end
                HRV_frequency_each["subset_duration"] = subset_duration
                HRV_frequency_each["Include"] = HRV_frequency_each.apply(inclusion_criteria,  axis=1)

                HRV_frequency_each["Series"] = run
                
                HRV_frequency = pd.concat([HRV_frequency, HRV_frequency_each], axis = 0, ignore_index = True)
                # plot segments that need to be reviewed
                # if  HRV_frequency_each["Include"][0] =="REVIEW":
                #     # Create an info dictionary with peaks info
                #     plot_info = {"PPG_Peaks": fq_data,
                #             "sampling_rate": sampling_rate}  # Indices of peaks where condition is met
                    
                #     plot = nk.ppg_plot(quality_segment, info = plot_info, static=False) 
                    
                #     save_path = os.path.join(savefolder, "plots", "Frequency_domain", f"{participant}_{timepoint}_{run}.html")
                #     pyo.plot(plot, filename=save_path, auto_open=False)

                ###############################################################################
                #####################Process Time Domain#######################################
                ###############################################################################

                ## create 10s slinding window epochs9
                if len(preprocess)< 10*sampling_rate: #skip if less than 10s of data
                    continue

                event_onsets = list(range(0, len(preprocess)-(sampling_rate*10), sampling_rate))
                event_durations = [sampling_rate * 10] * len(event_onsets)
                events = nk.events_create(event_onsets=event_onsets, event_durations= event_durations, event_labels=None, event_conditions=None)       
                epochs = nk.epochs_create(preprocess, events, sampling_rate=sampling_rate)

                ## extract peaks per epoch and process TD HRV
                hrv_time = pd.DataFrame()
                for num, epoch in epochs.items():
                    peaks = np.where(epoch["PPG_Peaks"])[0]
                    
                    if len(peaks)>2: # minimum of 3 peaks per segment
                        hrv_time_row = nk.hrv_time(peaks, sampling_rate = sampling_rate)
                    else:
                        hrv_time_row = pd.DataFrame()
                        hrv_time_row = hrv_time_row.append(pd.Series(), ignore_index=True)
                    ##add columns to dataframe row
                    hrv_time_row["Label"] = num
                    ### add quality check columns
                    hrv_time_row = add_quality_columns(hrv_time_row, epoch, below_threshold_index, TD = True, sampling_rate = sampling_rate)
                    hrv_time_row["Include"] = hrv_time_row.apply(inclusion_criteria, axis=1, TD=True)
                    ###save to results dataframe
                    hrv_time = pd.concat([hrv_time, hrv_time_row], axis = 0, ignore_index = True)  
                    ###save dataframe 
                    TD_path = os.path.join(savefolder, "whole_run", f"{participant}_{timepoint}_{run}_Time_Domain.csv")
                    hrv_time.to_csv(TD_path)

                    ## Plot segments with bordeline quality, primary conditions listed below
                    if len(hrv_time_row) > 0:
                        if  hrv_time_row["Include"][0] == "REVIEW":
                            
                            # Create an info dictionary with peaks info
                            epoch_new_index = epoch.reset_index()
                            
                            plot_info = {"PPG_Peaks": peaks,
                                    "sampling_rate": sampling_rate}  # Indices of peaks where condition is met
                            
                            plot = nk.ppg_plot(epoch_new_index, info = plot_info, static=False) 
                            
                            save_path = os.path.join(savefolder, "plots", "ppg_time", f"{participant}_{timepoint}_{run}_start_time_{num}.html")
                            pyo.plot(plot, filename=save_path, auto_open=False)
                    
                # Log Time Domain HRV quality
                log["Whole_run"] = log.get("Whole_run", {})
                log["Whole_run"]["Time_domain"] = log["Whole_run"].get("Time_domain", {})
                log["Whole_run"]["Time_domain"][f"{run}"] = log["Whole_run"]["Time_domain"].get(f"{run}", {})

                
                inlcuded_proportion = int((hrv_time["Include"] == True).sum())/len(hrv_time)
                For_Review_Count = int((hrv_time["Include"] == "REVIEW").sum())
                
                log["Whole_run"]["Time_domain"][f"{run}"]["Included_Proportion"] = inlcuded_proportion
                log["Whole_run"]["Time_domain"][f"{run}"]["For_Review"] = For_Review_Count


                ############################################################################
                ##############TASK RELATED ANALYSIS########################################
                ############################################################################
                
                ##identify task files for event related analysis
                if taskfiles.empty or run not in task_dict:
                    continue

                if len(task_dict[run]) == 1:
                    run_taskfile = task_dict[run][0]

                if len(task_dict[run]) > 1:
                    print(f"Multiple task files for {run} in {participant} {timepoint}")
                    exit 
                
                # taskfile_bool = taskfiles.str.contains(run, case = False)
                # taskfile_index = np.where(taskfile_bool)[0]

                # if len(taskfile_index) == 1:
                #     # Extract the single match as a string
                #     run_taskfile = taskfiles.iloc[taskfile_index[0]]
                # else:
                #     continue    

                # Proceed to create the paths

                #if city == "London": 
                tasks_path = os.path.join("Data", city, participant, timepoint, "Task", run_taskfile)
                # else:
                #     tasks_path = os.path.join("Data", city, participant, timepoint,"Task", run_taskfile)
                
                #create task info dataframe
                task, ratings = format_taskfile(tasks_path)

                Ratings_all[run] = ratings

        
                #Process events
                ## skip if resting scan
                to_process =["Run1", "Run2", "Run3"]

                if run not in to_process:
                    quality_df.to_csv(os.path.join(savefolder, "beat_quality", f"{participant}_{run}_{timepoint}_beat_quality.csv"))
                    continue
                    
            
                new_row = {
                    'TrialNo': 0, 
                    'TimeAtStartOfTrial': -8000, 
                    'TrialType': "Rest",
                    'TraumaCue': "None",
                    'TrialDuration': 8000,
                }
                
                task = task.append(new_row, ignore_index = True)
                #reorder task based on TrialNo
                task = task.sort_values(by = 'TimeAtStartOfTrial')
                task['TimeAtStartOfTrial'] = task['TimeAtStartOfTrial'] + 8000
                task = task.reset_index(drop = True)

                #create dicts of task info and lists of trials by type


                trial_info = get_trial_info(task, sampling_rate)
                
                No_trial = task.loc[task["IsOwn"] == "No", "TrialNo"].tolist()
                Yes_trial = task.loc[task["IsOwn"] == "Yes", "TrialNo"].tolist()
                
                #remove Nan values from trial info dict
                if pd.isna(trial_info["onset"][-1]) or pd.isna(trial_info["duration"][-1]):
                    for key in trial_info:
                        trial_info[key] = trial_info[key][:-1]

                #create epochs per trial 
                event_epochs = nk.epochs_create(preprocess_all, trial_info, sampling_rate = sampling_rate)
                #combine trials and save
                combined_trials = combine_epochs(event_epochs)
                combined_trials_path = os.path.join(savefolder,"event_related", f"{participant}_{run}_{timepoint}_preprocessed.csv")
                combined_trials.to_csv(combined_trials_path)
                

                # Check for duplicates in the 'Index' column
                duplicates = combined_trials[combined_trials.duplicated('Index', keep=False)]
                # If you want to remove duplicates, you can do so by keeping the first occurrence
                combined_trials_unique = combined_trials.drop_duplicates(subset='Index', keep='first')
                # Now you can safely create the value_map
                value_map = combined_trials_unique.set_index("Index")["Label"]

                # Map the values to the 'Trial_No' column in quality_df
                quality_df['Trial_No'] = quality_df["peak_index"].map(value_map)
                quality_df["Trial_No"] = pd.to_numeric(quality_df["Trial_No"], errors="coerce")
                quality_df["Own_Trial"] = quality_df["Trial_No"].isin(Yes_trial)
                ## save beat quality df
                quality_df.to_csv(os.path.join(savefolder, "beat_quality", f"{participant}_{run}_{timepoint}_beat_quality.csv"))

                #process TD HRV
                hrv_event = pd.DataFrame()
                acceleration = {}
                processed_IBI_run = []

                for trial_no, event in event_epochs.items():
                    if event.empty:
                        continue
                    
                    epoch_quality = np.mean(event["PPG_Quality"])
                    event_peaks = np.where(event["PPG_Peaks"])[0]
                    
                    if epoch_quality > quality_threshold: 
                        epoch_hrv_event = event
                        peaks_hrv_event = event_peaks
                        subset_used = "Not needed"
                        subset_start = np.nan
                        subset_end = np.nan
                        subset_duration = np.nan
                        
                    else: #find segment to use if event is below_threshold
                        min_duration = sampling_rate * 10 # 10S
                        quality_array = np.array(event["PPG_Quality"]) # to be iterated through

                        iteration_durations = list(range(min_duration, len(quality_array), sampling_rate))
                        iteration_starts = list(range(0, len(quality_array)-min_duration, sampling_rate))

                        longest_duration = 0
                        longest_start = 0

                        for duration in iteration_durations:
                            for start in iteration_starts:
                                # Check if there's enough data for this segment
                                if start + duration > len(quality_array):
                                    continue
                
                                # Compute the segment and its average
                                segment = quality_array[start : start + duration]
                                segment_average = np.mean(segment)
                            
                                # Check if this segment satisfies the condition
                                if segment_average > quality_threshold:
                                    longest_duration = int(duration)
                                    longest_start = start
                                    break
                                        
                        #check if a subsection was identified
                        if longest_duration > 0:
                            ## segment data according to quality checks
                            epoch_hrv_event = event.iloc[longest_start : longest_start + longest_duration]
                            epoch_quality = np.mean(epoch_hrv_event["PPG_Quality"])
                            subset_used = True
                            subset_start = longest_start
                            subset_end = longest_start + longest_duration
                            subset_duration = longest_duration
                        else:
                            epoch_hrv_event = event
                            subset_used = "None avaliable"
                            subset_start = np.nan
                            subset_end = np.nan
                            subset_duration = np.nan

                    ## process HRV
                    peaks_hrv_event = np.where(epoch_hrv_event["PPG_Peaks"])[0]
                    len_hrv_event = len(epoch_hrv_event)/sampling_rate
                    
                    if len(peaks_hrv_event)>2 and len(peaks_hrv_event) > ((len_hrv_event/5)+2): # minimum of 3 peaks per segment and baseline function from Rapid HRV toolbox
                        hrv_event_row = nk.hrv_time(peaks_hrv_event, sampling_rate = sampling_rate)
                    else:
                        hrv_event_row = pd.DataFrame()
                        hrv_event_row = hrv_event_row.append(pd.Series(), ignore_index=True)

                    ##add columns to dataframe row
                    hrv_event_row["Trial_No"] = trial_no
                    if int(trial_no) == 0:
                        hrv_event_row["Trial_Type"] = "NA"
                    else:
                        hrv_event_row["Trial_Type"] = "Own" if trial_no in Yes_trial else "Control"
                    hrv_event_row["subset_used"] = subset_used
                    hrv_event_row["subset_start"] = subset_start
                    hrv_event_row["subset_end"] = subset_end
                    hrv_event_row["subset_duration"] = subset_duration
                    hrv_event_row["Nowness"] = trial_info["Nowness"][int(trial_no)]
                    
                    
                    ### add quality check columns
                    hrv_event_row = add_quality_columns(hrv_event_row, epoch_hrv_event, below_threshold_index, TD=True, sampling_rate = sampling_rate)  
                    hrv_event_row["Include"] = hrv_event_row.apply(inclusion_criteria, axis=1, TD=True)
                    
                    ###save to results dataframe
                    hrv_event = pd.concat([hrv_event, hrv_event_row], axis = 0, ignore_index = True)  


                    ## Plot segments with bordeline quality, primary conditions listed below
                    if len(hrv_event_row) > 0:
                        if  hrv_event_row["Include"][0] == "REVIEW":
                            
                            # Create an info dictionary with peaks info
                            epoch_new_index = epoch_hrv_event.reset_index()
                            
                            plot_info = {"PPG_Peaks": peaks_hrv_event,
                                    "sampling_rate": sampling_rate}  # Indices of peaks where condition is met
                            
                            plot = nk.ppg_plot(epoch_new_index, info = plot_info, static=False) 
                            save_path = os.path.join(savefolder, "plots", "PPG_event", f"{participant}_{timepoint}_{run}_trial_No_{trial_no}.html")
                            pyo.plot(plot, filename=save_path, auto_open=False)
                    

                    #IBI ANALYSIS
                    
                ibi_trial_info = ibi_trial_info_format(trial_info, sampling_rate)
                ibi_event_epochs = nk.epochs_create(preprocess_all, ibi_trial_info, sampling_rate = sampling_rate)
                
                ibi_long_df = pd.DataFrame()

                for trial_no, event in ibi_event_epochs.items():
                    event_peaks = np.where(event["PPG_Peaks"])[0]
                    # subtract 4 seconds in sample indices from event_peaks
                    event_peaks = event_peaks - (4*sampling_rate)

                    ibi = [(event_peaks[i+1] - event_peaks[i])/sampling_rate for i in range(len(event_peaks)-1)]
                    ibi = np.array(ibi)

                    #check if IBI analysis should run
                    if len(event_peaks) > 2 and  (ibi < 3).all() and event_peaks[1] < 0 and event_peaks[-1] > 0:#no missed peaks & there is an ibi prior to trial start
                        last_pre_event_peak = max(np.where(event_peaks<0)[0])
                        cut_ibi = ibi[last_pre_event_peak:] # IBI start from last pre event ibi
                        normalised_to = cut_ibi[0] # first IBI before trial start
                        ibi_for_normalise = ibi[last_pre_event_peak - 3:last_pre_event_peak]
                        normalised_avg = np.mean(ibi_for_normalise)
                        ibi_final = np.insert(cut_ibi, 0, normalised_avg)
                        acceleration_event = (ibi_final/ normalised_avg) * 100 # normalise to first IBI before trial start and convert to percentage
                        acceleration[trial_no] = acceleration_event
                        processed_IBI_run.append(trial_no)

                        #create dataframe out of acceleration_event
                        acceleration_event_df = pd.DataFrame(acceleration_event, columns = ["Acceleration"])
                        acceleration_event_df["Trial_No"] = trial_no
                        acceleration_event_df["Trial_Type"] = "Own" if trial_no in Yes_trial else "Control"
                        acceleration_event_df["Nowness"] = trial_info["Nowness"][int(trial_no)]
                        acceleration_event_df["Beat_No"] = range(-1,len(acceleration_event_df)-1)

                        ibi_long_df = pd.concat([ibi_long_df, acceleration_event_df], axis = 0, ignore_index = True)

                #save IBI data
                ibi_long_df_path = os.path.join(savefolder, "event_related", f"{participant}_{run}_{timepoint}_IBI_long.csv")
                ibi_long_df.to_csv(ibi_long_df_path)

                if acceleration:
                    Yes_data, No_data = acceleration_by_trial(acceleration, Yes_trial, No_trial)
                    
                    #add means sem and sample present to dataframes 
                    Yes_data = mean_sem_samplepresent(Yes_data)
                    No_data = mean_sem_samplepresent(No_data)

                    Yes_filename = os.path.join(savefolder, "event_related", f"acceleration_yes_{participant}_{run}_{timepoint}.csv")
                    Yes_data.to_csv(Yes_filename)

                    No_filename = os.path.join(savefolder, "event_related", f"acceleration_no_{participant}_{run}_{timepoint}.csv")
                    No_data.to_csv(No_filename)
                
                    #create and save IBI plot
                    plotfolder = os.path.join(savefolder, "plots", "IBI", f"IBI_{participant}_{timepoint}_{run}")
                    if (Yes_data["sample_present"].any() and No_data["sample_present"].any()):
                        IBI_plot(Yes_data, No_data, plotfolder) 

                
                #log event related analysis  
                
                log["Event_Related"] = log.get("Event_Related", {})
                log["Event_Related"][f"{run}"] = log["Event_Related"].get(f"{run}", {})
                log["Event_Related"][f"{run}"] = add_to_log(log["Event_Related"][f"{run}"], hrv_event)

                ###save dataframe 
                event_TD_path = os.path.join(savefolder, "event_related", f"{participant}_{timepoint}_{run}_HRV.csv")
                hrv_event.to_csv(event_TD_path)
                
                
            #save ratings
            ratings_save_path = os.path.join(savefolder, f"ratings_{participant}_{timepoint}.json")
            with open(ratings_save_path, "w") as f:
                json.dump(Ratings_all, f, indent=4)

            # Add to FD dictionary
            log["Whole_run"] = log.get("Whole_run", {})
            log["Whole_run"]["Frequency_Domain"] = log["Whole_run"].get("Frequency_Domain", {})
            if HRV_frequency.empty:
                log["Whole_run"]["Frequency_Domain"] = "No data"
            else:   
                log["Whole_run"]["Frequency_Domain"] = add_to_log(log["Whole_run"]["Frequency_Domain"], HRV_frequency)

            #save FD for all runs 
            FD_path = os.path.join(savefolder, "whole_run", f"{participant}_{timepoint}_Frequency_Domain.csv")
            HRV_frequency.to_csv(FD_path)
            # Save the log
            savelog = os.path.join(savefolder, f"Processing_log_{participant}_{timepoint}.json")
            with open(savelog, "w") as f:
                json.dump(log, f, indent=4)

        #add participant number to processed
        Processed.append(participant)

            ## IBI check it first IBI is in first 3 seconds otherwise assumed missed beat
            #save frequency domain at end of runs
            
