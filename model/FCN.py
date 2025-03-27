'''
Author: syu-kylin 79462188+syu-kylin@users.noreply.github.com
Date: 2024-07-22 20:05:32
LastEditors: syu-kylin
LastEditTime: 2024-07-22 20:44:58
FilePath: \4d-HIE-HRV Delphi\model\FCN.py
Description: 

Copyright (c) 2024 by ${git_email}, All Rights Reserved.
'''



from torch import nn

class FCN13_HRV_5min(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # F1
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=4, stride=3),

            # F2
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=4, stride=3),

            # F3
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=4, stride=3),
            
            # F4
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm1d(32), nn.AvgPool1d(kernel_size=4, stride=3),
            
            # Classifier
            nn.Conv1d(32, 2, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Flatten()
            )
        
    def forward(self, x):
        '''Forward pass'''
        return self.network(x) 
        