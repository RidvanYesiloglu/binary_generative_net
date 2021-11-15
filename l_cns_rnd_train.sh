read -p "Write the objective function no (MeMeAC, MaMeAC, MEWSD, MEWSD_d, MF, MF_d, ELW): " obj_no
read -p "Use weight decay? (0: no, 1: yes) " wd_exists
read -p "Write run name: " run_name

mc_samps=500
eps=500000
no_runs=10
for K in 3 7 15 31; do 
    for N in 64 128 512 1024; do
        echo $K,$N
        sbatch ./exec.sh $K $N $obj_no cns_ini $run_name 0 $mc_samps $eps $no_runs
        sbatch ./exec.sh $K $N $obj_no rnd_ini $run_name 0 $mc_samps $eps $no_runs
    done
done
#echo "tail -n100 -f data/obj${obj_no}/sigm_bern_nes_model/${K}_${L}/init${init_mode}_sgs1.0_lr0.01_eps${eps}_smps${mc_samps}_run${run_name}/main_logs.txt"
