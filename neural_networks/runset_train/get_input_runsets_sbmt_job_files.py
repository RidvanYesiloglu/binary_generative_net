from runset_train import get_input_runsets_crt_opts
#from runset_train import submit_job_files_given_opts
import os
def main():
    dict_file = "params_dictionary"
    print("1 burasi")
    opt_strs, params_dict = get_input_runsets_crt_opts.main(dict_file)
    print("2 burasi")
    print(opt_strs)
    submit_job_files_given_opts.main(opt_strs.split("\n"), params_dict)
    print("3 burasi")

if __name__ == "__main__":
    main()
