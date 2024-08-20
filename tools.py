import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import  splrep, splev
import scipy.stats as st
import scipy.signal

R = 6377.726    # en km
pi = np.pi

import torch

def calculate_step_length(positions_batch):

    latitudes_batch = positions_batch[:, :, 0]
    longitudes_batch = positions_batch[:, :, 1]
    # Calculate the differences in latitude and longitude
    dlat_batch = latitudes_batch[:, 1:] - latitudes_batch[:, :-1]
    dlon_batch = longitudes_batch[:, 1:] - longitudes_batch[:, :-1]

    step_lengths_batch = torch.zeros((latitudes_batch.shape))

    # step_lengths_batch[:, 1:] = torch.sqrt(dlon_batch ** 2 + dlat_batch ** 2)
    step_lengths_batch[:, 1:] = dlon_batch ** 2 + dlat_batch ** 2

    # Add a 0 as the first value for each graph
    # step_lengths_batch = torch.cat((torch.zeros((step_lengths_batch.shape[0], 1)), step_lengths_batch), dim=1)

    return step_lengths_batch

def calculate_dist_nest0(positions_batch):

    nest = (positions_batch[:, 0, :] + positions_batch[:, 1, :]) / 2
    # print(nest.shape)
    # print(nest)
    for i in range(positions_batch.shape[0]):
        for j in range(positions_batch.shape[1]):
            positions_batch[i, j, :] = positions_batch[i, j, :] - nest[i, :]
    latitudes_batch = positions_batch[:, :, 0]
    longitudes_batch = positions_batch[:, :, 1]

    # dist_nest0 = torch.sqrt(latitudes_batch ** 2 + latitudes_batch ** 2)
    dist_nest0 = latitudes_batch ** 2 + latitudes_batch ** 2

    return dist_nest0


def dist_ortho(lon1, lat1, lon2, lat2):
    a = np.sin((lat1 - lat2)/2*np.pi/180)**2
    b = np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)
    c = np.sin((lon1- lon2)/2* np.pi/180)**2

    dist = R * 2* np.arcsin( np.sqrt(a + b*c))
    return dist

def dist_eucl(lon1, lat1, lon2, lat2):
    return np.sqrt((lon1-lon2)**2 + (lat1-lat2)**2)


def cap(lon1, lat1, lon2, lat2):
    # to radians
    lat1 = lat1*pi/180
    lat2 = lat2*pi/180
    lon1 = lon1*pi/180
    lon2 = lon2*pi/180

    delta_lon = lon2-lon1

    a = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(delta_lon)
    b = np.sin(delta_lon) * np.cos(lat2)

    cap = np.arctan2(b , a)
    cap = cap%(2*np.pi)

    return cap*180/np.pi


def get_dist_nest(colony, traj):
    dist_nest = []
    for i in range(len(traj[:, 0])) :
        dist = dist_ortho(colony[0], colony[1], traj[i, 0], traj[i, 1])
        dist_nest.append(dist)
    return np.array(dist_nest)


def get_max_dist_nest(colony, traj):
    return np.max(get_dist_nest(colony, traj))


def get_step_length(traj):
    return np.concatenate([[0], dist_ortho(traj[:,0][:-1], traj[:,1][:-1], traj[:,0][1:], traj[:,1][1:])])

def total_distance(traj):
    return np.sum(get_step_length(traj))



def angle_changes(traj):
    vectors = np.diff(traj, axis=0)
    angles = np.arctan2(vectors[:,1], vectors[:,0])
    angle_changes = np.diff(angles)
    return angle_changes


def get_trip_sinuosity(df):
    return 2*(np.max(df[:, 3]) - np.min(df[:, 3])) /np.sum(df[:, 2])


def kde1d(z, bw=2):
    kernel = st.gaussian_kde(z)
    kernel.set_bandwidth(bw_method=bw)
    x = np.arange(min(z), max(z), 1)
    return kernel

def cap(lon1, lat1, lon2, lat2):
    # to radians
    lat1 = lat1*pi/180
    lat2 = lat2*pi/180
    lon1 = lon1*pi/180
    lon2 = lon2*pi/180

    delta_lon = lon2-lon1

    a = np.cos(lat1) * np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(delta_lon)
    b = np.sin(delta_lon) * np.cos(lat2)

    cap = np.arctan2(b , a)
    cap = cap%(2*pi)

    return cap*180/pi