#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Mar  2 22:15:21 2020

@author: linux-asd
"""
import pybullet as p
import time
import numpy as np
import sys


class pybulletDebug:

  def __init__(self, control=[0.0, 0.0, 0.0]):
    #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains
    # on the robot)
    self.cyaw = 45
    self.cpitch = -20
    self.cdist = 2

    # up/down, left/right
    self.control = control

    time.sleep(0.5)

  def cam_and_robotstates(self, boxId):
    robotPos, robotOrn = p.getBasePositionAndOrientation(boxId)
    p.resetDebugVisualizerCamera(
        cameraDistance=self.cdist, cameraYaw=self.cyaw,
        cameraPitch=self.cpitch, cameraTargetPosition=robotPos
    )
    keys = p.getKeyboardEvents()
    #Keys to change camera
    if keys.get(100):  #D
      self.cyaw += .5
    if keys.get(97):  #A
      self.cyaw -= .5
    if keys.get(99):  #C
      self.cpitch += .5
    if keys.get(102):  #F
      self.cpitch -= .5
    if keys.get(122):  #Z
      self.cdist += .02
    if keys.get(120):  #X
      self.cdist -= .02
    
    if self.control is not None:
      if keys.get(65309): # Enter Key
        self.control = [0.0, 0.0, 0.0]
      if keys.get(65297):  # Right Arrow
        self.control[0] = np.clip(self.control[0] + 0.1, -2.0, 2.0)
      if keys.get(65298):  # Left Arrow
        self.control[0] = np.clip(self.control[0] - 0.1, -2.0, 2.0)
      if keys.get(65296):  # Up Arrow
        self.control[1] = np.clip(self.control[1] + 0.1, -3.0, 3.0)
      if keys.get(65295):  # Down Arrow
        self.control[1] = np.clip(self.control[1] - 0.1, -3.0, 3.0)
      if keys.get(50):  # 2 Key
        self.control[2] = np.clip(self.control[2] - 0.05, -0.5, 0.5)
      if keys.get(51):  # 3 Key
        self.control[2] = np.clip(self.control[2] + 0.05, -0.5, 0.5)
    
    if keys.get(27):  #ESC
      p.disconnect()
      sys.exit()
  
  def get_action(self):
    return np.array(self.control)