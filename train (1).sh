read -p "Write the number of codes (satellites): " K
read -p "Write the period of codes: " L
read -p "Write the objective function no (MeMeAC, MaMeAC, MEWSD, MEWSD_d, MF, MF_d, ELW): " obj_no
read -p "Which init mode to use? (cns_ini, rnd_ini, gold_ini, weil_ini, ) " init_mode
read -p "Use weight decay? (0: no, 1: yes) " wd_exists
read -p "Write run name: " run_name
mc_samps=500
eps=50
no_runs=2
srun ./exec.sh $K $L $obj_no $init_mode $run_name $wd_exists $mc_samps $eps $no_runs
echo "tail -n100 -f data/obj${obj_no}/sigm_bern_nes_model/${K}_${L}/init${init_mode}_sgs1.0_lr0.01_eps${eps}_smps${mc_samps}_run${run_name}/main_logs.txt"