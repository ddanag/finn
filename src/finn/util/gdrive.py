# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gspread
import os
import warnings
from datetime import datetime
import time

def upload_to_end2end_dashboard(data_dict):
    gdrive_key = "/workspace/finn/gdrive-key/service_account.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
    gc = gspread.service_account(filename=gdrive_key)
    spreadsheet = gc.open("finn-end2end-dashboard")
    worksheet = spreadsheet.get_worksheet(0)
    keys = list(data_dict.keys())
    vals = list(data_dict.values())
    # check against existing header
    existing_keys = worksheet.row_values(1)
    if not set(existing_keys).issuperset(set(keys)):
        # create new worksheet
        dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet = spreadsheet.add_worksheet(
            title="Dashboard " + dtstr, rows=10, cols=len(keys), index=0
        )
        # create header row with keys
        worksheet.update("A1:1", [keys])
        # freeze and make header bold
        worksheet.freeze(rows=1)
        worksheet.format("A1:1", {"textFormat": {"bold": True}})
    # insert values into new row at appropriate positions
    worksheet.insert_row([], index=2)
    for i in range(len(keys)):
        colind = existing_keys.index(keys[i])
        col_letter = chr(ord("A") + colind)
        worksheet.update("%s2" % col_letter, vals[i])

def create_worksheet_in_resource_dashboard(worksheet_name, no_of_rows, no_of_columns):
    gdrive_key = "/workspace/finn/gdrive-key/finn-gdrive-key.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
    gc = gspread.service_account(filename=gdrive_key)
    spreadsheet = gc.open("finn-resource-dashboard")
    spreadsheet.add_worksheet(title=worksheet_name, rows=no_of_rows, cols=no_of_columns)

    #DD should add 'add_headers' option

def delete_worksheet_from_resource_dashboard(worksheet_name):
    gdrive_key = "/workspace/finn/gdrive-key/finn-gdrive-key.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
    gc = gspread.service_account(filename=gdrive_key)
    spreadsheet = gc.open("finn-resource-dashboard")
    spreadsheet.del_worksheet(spreadsheet.worksheet(worksheet_name))

def get_records_from_resource_dashboard(worksheet_name):
    #returns list of dicts
    gdrive_key = "/workspace/finn/gdrive-key/finn-gdrive-key.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
    
    #in case of usage limit error, waits a second and tries to access the dashboard again
    n = 0
    while n <= 60:
        try:
            gc = gspread.service_account(filename=gdrive_key)
            spreadsheet = gc.open("finn-resource-dashboard")
            worksheet = spreadsheet.worksheet(worksheet_name)
            break    
        except:
            n = n + 1
            time.sleep(1) 

    list_of_dicts = worksheet.get_all_records()

    return list_of_dicts

def search_in_resource_dashboard(worksheet_name, data_dict):
    """Looks for the contents of data_dict in the worksheet,
    returns True and row index if matched, False and -1 otherwise.
    "get_all_records" returns list of dicts - headers (which are the keys of the dict)
    and row values. If data_dict is a subset of a certain dict in that list, matched is
    returned as True.
    Format: data_dict = {header1 : value1, header2 : value2, etc}"""

    matched = False
    row_index = -1
    n = 0

    gdrive_key = "/workspace/finn/gdrive-key/finn-gdrive-key.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
        
    while n <= 60:
        try:
            gc = gspread.service_account(filename=gdrive_key)
            spreadsheet = gc.open("finn-resource-dashboard")
            worksheet = spreadsheet.worksheet(worksheet_name)
            break    
        except:
            n = n + 1
            time.sleep(1) 

    #in case of usage limit error, waits a second and tries to get the records again
    while n <= 60:
        try:
            list_of_dicts = worksheet.get_all_records()
            for dictionary in list_of_dicts:
                row_index = list_of_dicts.index(dictionary) + 2
                dictionary = {str(key): str(value) for key, value in dictionary.items()}
                data_dict = {str(key): str(value) for key, value in data_dict.items()}
                if data_dict.items() <= dictionary.items():
                    matched = True
                    break
                else:
                    row_index = -1
            break    
        except:
            n = n + 1
            time.sleep(1)

    return matched, row_index

def upload_to_resource_dashboard(worksheet_name, data_dict, overwrite, row_index):
    
    gdrive_key = "/workspace/finn/gdrive-key/finn-gdrive-key.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return

    n = 0 
    while n <= 60:
        try:
            gc = gspread.service_account(filename=gdrive_key)
            spreadsheet = gc.open("finn-resource-dashboard")
            worksheet = spreadsheet.worksheet(worksheet_name)
            break    
        except:
            n = n + 1
            time.sleep(1)

    keys = list(data_dict.keys())
    vals = list(data_dict.values())

    """ should remove this, increases the gspread number of requests
    #check if the worksheet exists, if not create one
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)   
    except:    
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=10, cols=len(keys))
        # create header row with keys
        worksheet.update("A1:1", [keys])
        # freeze and make header bold
        worksheet.freeze(rows=1)
        worksheet.format("A1:1", {"textFormat": {"bold": True}})
    """
    while n <= 60:
        try:
            existing_keys = worksheet.row_values(1)
            if overwrite:
                #update values at row_index
                spreadsheet.values_update('%s!A%s:AZ%s' % (worksheet_name, row_index, row_index), params={'valueInputOption': 'USER_ENTERED'}, body={'values': [vals]})
            elif not overwrite:
                # insert values into new row
                spreadsheet.values_append('%s!A1:AZ1' % worksheet_name, params={'insertDataOption': 'INSERT_ROWS', 'valueInputOption': 'USER_ENTERED'}, body={'values': [vals]})
            break    
        except:
            n = n + 1
            time.sleep(1)
            