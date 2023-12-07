import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker

import time
import matplotlib.dates as md
import dateutil
from tqdm import tqdm
import argparse


def max_end_time(data):
    max_end_time = 0
    length = len(data['END_TIMESTAMP'])
    for i in tqdm(range(0,length)):
        end_time = md.date2num(dateutil.parser.parse(data['END_TIMESTAMP'][i]))
        if max_end_time < end_time:
            max_end_time = end_time
    return max_end_time


def sort_job_id(data, sort_time):
    job_tuples = []
    length = len(data[sort_time])
#   length = 5
    for i in tqdm(range(0,length)):
        job_tuples.append((i,md.date2num(dateutil.parser.parse(data[sort_time][i]))))
#   print(job_tuples)
    new_job_tuples = sorted(job_tuples, key=lambda time: time[1])
#   print(new_job_tuples)
    job_sorted_list = np.zeros([length], np.int)
    for i in tqdm(range(0,length)):
        job_sorted_list[i] = new_job_tuples[i][0]
    return job_sorted_list


def Num_of_Jobs_per_time_slot(data, time_slot, time, sorted_map, end_time, start_job_idx, end_job_idx):
    print(dateutil.parser.parse(data[time][sorted_map[start_job_idx]]))
    start_time = md.date2num(dateutil.parser.parse(data[time][sorted_map[start_job_idx]]))
    total_time = end_time - start_time
    num_slots = int(total_time / time_slot) + 1
    max_end_time = start_time
    
    num_jobs_per_slot = np.zeros([num_slots],np.int64)
    num_parallel_jobs_per_slot = np.zeros([num_slots],np.int64)
    num_ratio_per_slot = np.zeros([num_slots],np.float64)
    time_slots = np.zeros([num_slots],np.float64)
    
    num_jobs = end_job_idx - start_job_idx
#   num_jobs = 10
#   for i in tqdm(range(start_job_idx, end_job_idx)):
    for i in range(start_job_idx, end_job_idx):
        start = md.date2num(dateutil.parser.parse(data[time][sorted_map[i]]))
        end = md.date2num(dateutil.parser.parse(data['END_TIMESTAMP'][sorted_map[i]]))
        if (max_end_time < end):
            max_end_time = end
        parallel_jobs = data['NUM_TASKS_MULTILOCATION'][sorted_map[i]]
        start_index = int((start-start_time)/time_slot)
        end_index = int((end - start_time )/time_slot) + 1
#    print(start,end,start_index,end_index,end,start_time ,time_slot)
        for j in range(start_index, end_index):
            num_jobs_per_slot[j] += 1
            if parallel_jobs > 0:
                num_parallel_jobs_per_slot[j] +=1
    
    new_num_slots = int((max_end_time - start_time)/ time_slot) + 1
    for i in range(0, num_slots):
        time_slots[i] = start_time + i * time_slot + 0.5 * time_slot
        if(np.float64(num_jobs_per_slot[i]) > 0):
            num_ratio_per_slot[i] = np.float64(num_parallel_jobs_per_slot[i]) / np.float64(num_jobs_per_slot[i])
        else:
            num_ratio_per_slot[i] = 0
    
    return time_slots[:new_num_slots], num_parallel_jobs_per_slot[:new_num_slots], num_jobs_per_slot[:new_num_slots], num_ratio_per_slot[:new_num_slots], start_time, max_end_time
        

def TransferMidPlane(location, job_id):
    location_list = location.split("-")
    if(location_list[-1] == 'UNKNOWN'):
        return []
    num_nodes = int(location_list[-1])
    num_midplanes = num_nodes / 512
    midplane_list = []
    midplane_id = [0, 0, 0, 0, 0]
    start_node_ids = [0, 0, 0, 0, 0]
    end_node_ids = [0, 0, 0, 0, 0]
    k = 0
    start_node_ids[k] = int(location_list[1][k],16)
    end_node_ids[k] = int(location_list[2][k],16) + 1
    if (end_node_ids[k] - start_node_ids[k]) % 4 != 0:
        print("Error!! There is a job not use the whole midplane")
    for midplane_id[k] in range(start_node_ids[k], end_node_ids[k], 4):
        k = 1
        start_node_ids[k] = int(location_list[1][k],16)
        end_node_ids[k] = int(location_list[2][k],16) + 1
        if (end_node_ids[k] - start_node_ids[k]) % 4 != 0:
            print("Error!! There is a job not use the whole midplane")
        for midplane_id[k] in range(start_node_ids[k], end_node_ids[k], 4):
            k = 2
            start_node_ids[k] = int(location_list[1][k],16)
            end_node_ids[k] = int(location_list[2][k],16) + 1
            if (end_node_ids[k] - start_node_ids[k]) % 4 != 0:
                print("Error!! There is a job not use the whole midplane")
            for midplane_id[k] in range(start_node_ids[k], end_node_ids[k], 4):
                k = 3
                start_node_ids[k] = int(location_list[1][k],16)
                end_node_ids[k] = int(location_list[2][k],16) + 1
                if (end_node_ids[k] - start_node_ids[k]) % 4 != 0:
                    print("Error!! There is a job not use the whole midplane")
                for midplane_id[k] in range(start_node_ids[k], end_node_ids[k], 4):
#       print(midplane_id[k], start_node_ids[k], end_node_ids[k])
                    k = 4
                    start_node_ids[k] = int(location_list[1][k],16)
                    end_node_ids[k] = int(location_list[2][k],16) + 1
                    if (end_node_ids[k] - start_node_ids[k]) % 2 != 0:
                        print("Error!! There is a job not use the whole midplane")
                    for midplane_id[k] in range(start_node_ids[k], end_node_ids[k], 2):
                        new_midplane_id = midplane_id[3]/4 + midplane_id[2]/4 * 4 + midplane_id[1]/4 * 4 * 4 + midplane_id[0]/4 * 4 * 4 * 3
                        midplane_list.append(int(new_midplane_id))
#         print(new_midplane_id,midplane_id)
                    k = 3
                k = 2
            k = 1
        k = 0
    if len(midplane_list)*512 != num_nodes:
        print("[job id = %d]Error !! start_node_id = %s, end_node_id = %s, total = %d not equal to %s" %(job_id, location_list[1], location_list[2], len(midplane_list)*512, num_nodes))
        print(location)
    return midplane_list

def draw_job_distribution(data,start_job_id,end_job_id, max_end_time, Sorted_start_time_list):
    total_color = int(3)
    cmap = mpl.cm.get_cmap("rainbow", total_color )
    colors = cmap(np.linspace(0, 1, total_color ))
    time_slot = md.date2num(dateutil.parser.parse("2016-01-01 01:00:00")) - md.date2num(dateutil.parser.parse("2016-01-01 00:00:00"))
    
    start_time_slots, start_num_parallel_jobs_per_slot, start_num_jobs_per_slot, start_num_ratio_per_slot, min_time, max_time = Num_of_Jobs_per_time_slot(data, time_slot, 'START_TIMESTAMP', Sorted_start_time_list, max_end_time, start_job_id, end_job_id)
    num_figs = 1
    y_max = max_time/num_figs
    y_min = min_time
    total_length = float(y_max - y_min)
    total_y_ticks = 20
    y_tick_list = [y_min+total_length/total_y_ticks*i for i in range(0,total_y_ticks+1)]

    f, (ax) = plt.subplots(figsize=(30, 8))
    ax1 = ax.bar(start_time_slots, start_num_jobs_per_slot, color = colors[0], label = "total", width = 0.8*np.float64(10)/len(start_time_slots))
    ax2 = ax.bar(start_time_slots, start_num_parallel_jobs_per_slot, color = colors[1], label = "parallel", width = 0.8*np.float64(10)/len(start_time_slots))

    plt.xticks(y_tick_list,md.num2date(y_tick_list),rotation=90)
    yfmt = md.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(yfmt)
    plt.legend()
    plt.xlim([y_min,y_max])
    plt.ylabel("num of jobs")
    plt.xlabel("executed time")
    plt.title(f"Jobs ({start_job_id}, {end_job_id}) distribution between start time and end time\n total parallel jobs = {sum(start_num_parallel_jobs_per_slot)}, total jobs = {sum(start_num_jobs_per_slot)}, average = {np.float64(sum(start_num_parallel_jobs_per_slot))/sum(start_num_jobs_per_slot)}")
    plt.savefig(f"fig/start_time_{start_job_id}_{end_job_id}.png",bbox_inches = 'tight')
#   plt.show()

    f, (ax) = plt.subplots(figsize=(30, 8))
    ax.bar(start_time_slots, start_num_ratio_per_slot, color = colors[2], label = "ratio",  width = 0.8*np.float64(10)/len(start_time_slots))

    plt.xticks(y_tick_list,md.num2date(y_tick_list),rotation=90)
    yfmt = md.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(yfmt)
    plt.xlim([y_min,y_max])
    plt.ylabel("parallel jobs ratio")
    plt.xlabel("executed time")
    plt.title(f"Parallel job ratio ({start_job_id}, {end_job_id}) distribution between start time and end time\n averge parallel job ratio = {np.float64(sum(start_num_ratio_per_slot))/len(start_time_slots)}")
    plt.savefig(f"fig/start_time_ratio_{start_job_id}_{end_job_id}.png",bbox_inches = 'tight')
#   plt.show()
    
    f, (ax) = plt.subplots(figsize=(20, 8))

    total_midplanes = 96

    total_midplane_list = range(0,total_midplanes)

    total_color = int(30)
    cmap = mpl.cm.get_cmap("rainbow", total_color )
    colors = cmap(np.linspace(0, 1, total_color ))
    # print(colors)

    plt.ylim([0,total_midplanes])

    y_max = md.date2num(dateutil.parser.parse("2014-12-31 00:00:00"))
    y_min = md.date2num(dateutil.parser.parse("2016-01-01 00:00:00"))
    

#   for i in tqdm(range(start_job_id,end_job_id)):
    for i in range(start_job_id,end_job_id):
        job_id = Sorted_start_time_list[i]
#   print(job_id, data['JOB_NAME'][job_id])
        used_midplane_list = TransferMidPlane(data['LOCATION'][job_id],job_id)
        start_dates_in_each_midplane = []
        end_dates_in_each_midplane = []
        total_used_midplane_list = []
        start_time = md.date2num(dateutil.parser.parse(data['START_TIMESTAMP'][job_id]))
        end_time = md.date2num(dateutil.parser.parse(data['END_TIMESTAMP'][job_id]))
        if(y_min > start_time):
            y_min = start_time
        if y_max < end_time:
            y_max = end_time
#   print(start_time,end_time)
        for midplane_id in total_midplane_list:
            if midplane_id in used_midplane_list:
                total_used_midplane_list.append(midplane_id)
                start_dates_in_each_midplane.append(start_time)
                end_dates_in_each_midplane.append(end_time - start_time)
#   print(job_id, total_color , job_id % total_color )
        if(data['NUM_TASKS_MULTILOCATION'][job_id] == 0):
            ax.barh(y=total_used_midplane_list, left=start_dates_in_each_midplane, width=end_dates_in_each_midplane, color = colors[int(job_id%total_color)])
        else:
            ax.barh(y=total_used_midplane_list, left=start_dates_in_each_midplane, width=end_dates_in_each_midplane, color = colors[int(job_id%total_color)], ec='black', ls='-')

    total_length = float(y_max - y_min)
    total_y_ticks = 30
    y_tick_list = [y_min+total_length/total_y_ticks*i for i in range(0,total_y_ticks+1)]
    # print(y_tick_list)
    # print(md.num2date(y_tick_list))

    plt.yticks(range(0,96,5))
    plt.xticks(y_tick_list,md.num2date(y_tick_list),rotation=90)
    yfmt = md.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(yfmt)
    plt.xlim([y_min,y_max])
    plt.ylabel("midplanes")
    plt.xlabel("executed time")
    plt.title("Job Distribution")
    plt.savefig(f"fig/Job_distribution_{start_job_id}_{end_job_id}.png",bbox_inches = 'tight')
#   plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", help="the path of trace file")
    parser.add_argument("--num_jobs_per_fig", help="the number of jobs per figure")
    parser.add_argument("--start_job_id", help="the start job index")
    parser.add_argument("--end_job_id", help="the end job index")
    
    trace_path = "../traces/ANL-ALCF-DJC-MIRA_20150101_20151231.csv"
    num_jobs_per_fig = 2000
    start_job_id = 0
    end_job_id = 0
    args = parser.parse_args()
    if args.trace:
        trace_path = str(args.trace)
    if args.num_jobs_per_fig:
        num_jobs_per_fig = int(args.num_jobs_per_fig)
    if args.start_job_id:
        start_job_id = int(args.start_job_id)
    if args.end_job_id:
        end_job_id = int(args.end_job_id)

    data = pd.read_csv(trace_path)
    max_end_time = max_end_time(data)
    Sorted_start_time_list = sort_job_id(data,'START_TIMESTAMP')
    if end_job_id == 0:
        end_job_id = len(data['LOCATION'])
    
    
    for i in tqdm(range(start_job_id, end_job_id, num_jobs_per_fig)):
        start_job_id0 = i
        end_job_id0 = i + num_jobs_per_fig
        if end_job_id0 > end_job_id:
            end_job_id0 = end_job_id
        print(start_job_id0, end_job_id0)
        draw_job_distribution(data, start_job_id0, end_job_id0, max_end_time, Sorted_start_time_list)




