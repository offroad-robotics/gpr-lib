# Copyright (c) 2021, Jeremy Roy
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Offroad Robotics Lab at Queen's University nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file plots the ground truth for the 2D example
#
# Author: Jeremy Roy <jeremy.roy@queensu.ca>
# License: BSD 2.0

import matplotlib.pyplot as plt
import numpy as np

# Center of terrains element chuncks
bump_center  = np.array([ 2046.298,  1599.936]) # mm
grass_center = np.array([  236.904, -2487.764]) # mm
rocks_center = np.array([-1917.432,  1926.706]) # mm

# Corners of terrain element chuncks
bumps1 = (bump_center + np.array([ 441.689, -898.057])) / 1000 # m
bumps2 = (bump_center + np.array([-443.011, -900.151])) / 1000 # m
bumps3 = (bump_center + np.array([-442.218,  899.451])) / 1000 # m
bumps4 = (bump_center + np.array([ 443.539,  898.757])) / 1000 # m

grass1 = (grass_center + np.array([ 1805.520, -438.652])) / 1000 # m
grass2 = (grass_center + np.array([ 1804.230,  443.375])) / 1000 # m
grass3 = (grass_center + np.array([-1799.550,  438.900])) / 1000 # m
grass4 = (grass_center + np.array([-1810.200, -443.624])) / 1000 # m

rocks1 = (rocks_center + np.array([   -4.514,  1263.860])) / 1000 # m
rocks2 = (rocks_center + np.array([ 1259.800,    12.871])) / 1000 # m
rocks3 = (rocks_center + np.array([    8.747, -1267.760])) / 1000 # m
rocks4 = (rocks_center + np.array([-1264.030,    -8.969])) / 1000 # m

# Group corner markers of terrain elements for 2D plotting
bumps_2d = np.array([bumps1, bumps2, bumps3, bumps4, bumps1])
grass_2d = np.array([grass1, grass2, grass3, grass4, grass1])
rocks_2d = np.array([rocks1, rocks2, rocks3, rocks4, rocks1])


# Plot the wheelbase buffers
def plot_ground_truth(ax):
    # Plot terrain element markers
    ax.plot(bumps_2d.T[0], bumps_2d.T[1], "-", color="#ffb31a", label="Bumps")
    ax.plot(grass_2d.T[0], grass_2d.T[1], "-", color="#00802b", label="Grass")
    ax.plot(rocks_2d.T[0], rocks_2d.T[1], "-", color="black", label="Rocks")
