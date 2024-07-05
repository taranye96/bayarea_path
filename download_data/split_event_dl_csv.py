#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:44:09 2024

@author: tnye
"""

# Imports
import csv
import math

def split_csv(input_file, parts=9):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Extract the header
        header = next(reader)
        
        # Read all rows
        rows = list(reader)
        
        # Calculate number of rows per part
        rows_per_part = math.ceil(len(rows) / parts)
        
        for i in range(parts):
            start_index = i * rows_per_part
            end_index = start_index + rows_per_part
            part_rows = rows[start_index:end_index]
            output_file = f'/Users/tnye/bayarea_path/files/metadata/bayarea_catalog_2000-2024_libcomcat_{i}.csv'
            write_csv(output_file, header, part_rows)

def write_csv(output_file, header, rows):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Written {output_file}")

# Example usage
input_file = '/Users/tnye/bayarea_path/files/metadata/bayarea_catalog_2000-2024_libcomcat.csv'
split_csv(input_file)
