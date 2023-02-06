# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:19:35 2020

@author: jacqu

File to run vina docking on a dataframe with smiles strings in 'can' column 
    
(Paths are set for compute canada server and rupert. ) 

TODO : adapt to more general setting 

"""

import sys
import subprocess
import os 
import shutil
import argparse
from time import time
import numpy as np

#import openbabel
from openbabel import pybel
#import pybel 

import pandas as pd
  
def cline():
    # Parses arguments and calls main function with these args
    parser = argparse.ArgumentParser()
    
    ## Proteins
    parser.add_argument("-t", "--target", default='aa2ar', help="prefix of pdb receptor file. PDB file should be in data/receptors")
    
    ## Ligands 경로
    parser.add_argument("-df", "--dataframe_path", default='data/ligands/my_ligands.csv', 
                        help="Path to csv file with 'can' columns containing smiles")
    
    ## 계산 결과
    parser.add_argument("-o", "--out_dataframe_path", default='data/scored/my_ligands_scored.csv', 
                        help="Path to csv file for saving scores")
    
    ## 추가적인 경로 조정
    parser.add_argument("-s", "--server", default='rup', help="Server to run the docking on, for path and configs.")
    
    ## 파라미터 값 조정
    parser.add_argument("-e", "--ex", default=16, help="exhaustiveness parameter for vina. Default to 8")
    args = parser.parse_args()
    
    main(args)
    
def main(args):
    # Runs the docking process with the args provided
    
    ## sever 파라미터에 따른 추가적인 경로 조정
    if(args.server=='rup'):
        home_dir='/home/teamAIdrug/ybHong/Current_model_python/fine_tune_eval/Current'
        install_dir = '/home/teamAIdrug/ybHong/Current_model_python/fine_tune_eval/Current'
    elif(args.server=='cedar'):
        home_dir='/home/jboitr/projects/def-jeromew/jboitr'
        install_dir = '/home/jboitr/projects/def-jeromew/docking_setup'
    else:
        print('Error: "server" argument never used before. Set paths of vina/mgltools installs for this server.')
    
    # Uncomment to Copy receptor file from the DUDE dir if first time using this target. 
    #shutil.copyfile(f'/home/mcb/users/jboitr/data/all/{args.target}/receptor.pdb',f'data/receptors/{args.target}.pdb')
    
    ## Protein 경로
    #receptor_filepath = f'data/receptors/{args.target}.pdb'
    
    # target to pdbqt
    ## Protein 를 pdbqt로
    #subprocess.run(['python3','pdb_select.py',f'{receptor_filepath}','! hydro', f'{receptor_filepath}'])
    #subprocess.run([f'{install_dir}/mgltools_x86_64Linux2_1.5.6/bin/pythonsh', 'prepare_receptor4.py',
    #                f'-r {home_dir}/vina_docking/{receptor_filepath}','-o tmp/receptor.pdbqt', '-A hydrogens'])
    
    # Iterate on molecules
    ## Ligands 가져오기
    mols_df = pd.read_csv(args.dataframe_path)
    mols_df['score'], mols_df['time'] = pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    mols_list = mols_df['can']
    print("Docking molecules", {len(mols_list)})

    
    for i,smi in enumerate(mols_list):
        # smiles to mol2 
        SMILES_ERROR_FLAG=False
        with open('tmp/ligand.mol2', 'w') as f:
            try:
                mol = pybel.readstring("smi", smi)
                mol.addh()
                mol.make3D()
                
                txt = mol.write('mol2')
                f.write(txt)
                f.close()
                
            except:
                SMILES_ERROR_FLAG=True
                mean_sc=0.0
                delta_t=0.0
        
        #highest_sc = 0
        if(not SMILES_ERROR_FLAG):
            # ligand mol2 to pdbqt 
            ## Pybel을 통해서 ligand 검증 후 pdbqt로 변환
            subprocess.run(['{}/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'.format(install_dir), 'prepare_ligand4.py',
                            '-l tmp/ligand.mol2', '-o tmp/ligand.pdbqt']) #, '-A hydrogens'
            
            # RUN DOCKING
            ## 분석 시작
            start=time()
            subprocess.run(['{}/autodock_vina_1_1_2_linux_x86/bin/vina'.format(install_dir),
                        '--config', '{}/vina_docking/data/conf/conf_{}.txt'.format(home_dir, args.target),'--exhaustiveness', '{}'.format(args.ex), 
                        '--log', 'tmp/log.txt', '--seed', '42'])
            end = time()
            delta_t=end-start
            print("Docking time :", delta_t)

            if(delta_t>1): # Condition to check the molecule was docked 
                #reading output tmp/ligand_out.pdbqt
                with open('tmp/ligand_out.pdbqt','r') as f :
                    lines = f.readlines()
                    slines = [l for l in lines if l.startswith('REMARK VINA RESULT')]
                    #print(f'{len(slines)} poses found' )
                    values = [l.split() for l in slines]
                    # In each split string, item with index 3 should be the kcal/mol energy.
                    ## 저장할 Score의 Index 값을 지정해 줄 수 있음
                    #mean_sc=np.mean([float(v[3]) for v in values])
                    highest_sc=[v[3] for v in values][0]  # float(v[3])
                    #print([float(v[3]) for v in values][0])
            else:
                mean_sc=0.0
                
                            
        # Add to dataframe
        ## 점수와 걸린 시간을 Dataframe에 저장
        mols_df.at[i,'score']=highest_sc
        mols_df.at[i,'time']=delta_t
        
        ## 100번째 Ligand 마다 주기적으로 결과 값을 저장
        if(i%100==0): # checkpoint , save dataframe 
            mols_df.to_csv(args.out_dataframe_path)
            
    #final save 
    print('Docking finished, saving to csv')        
    mols_df.to_csv(args.out_dataframe_path)
    
if(__name__=='__main__'):
    cline()