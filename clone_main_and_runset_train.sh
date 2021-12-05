proj_name="GNSS_Code_Design_Project"
read -p "Do you want ssh permit (yes:1, no:0)? : " ssh_request
if (( ${ssh_request} == 1 ))
then
  ssh_key_file="../git_ssh_keys/id_ed25519"
  eval "$(ssh-agent -s)" # start ssh-agent
  chmod 400 $ssh_key_file # give yourself read permission for the ssh-key
  ssh-add $ssh_key_file # add the key to authorize you for cloning
  git init
  git pull git@github.com:RidvanYesiloglu/${proj_name}.git #clone
else
# Tara's pull command with git pull https here.
fi


cd ..
chmod +x ./GNSS_Code_Design_Project/neural_networks/runset_train.sh
./GNSS_Code_Design_Project/neural_networks/runset_train.sh #run runset_train.sh bash fil
