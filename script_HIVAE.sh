#source ~/venv/bin/activate

# declare dataset="Wine"
declare dataset="Letter"
declare batch_size=100

declare m_perc=20
declare mask=1

declare data_file=dataset/${dataset}/data.csv
declare types_file=dataset/${dataset}/data_types.csv
declare miss_file=dataset/${dataset}/Missing${m_perc}_${mask}.csv
declare true_miss_file=dataset/${dataset}/MissingTrue.csv


# declare model="HIVAE_inputDropout"
declare model="HIVAE_factorized"
declare z_dim=10
declare y_dim=5
declare s_dim=10


train_model(){
    sudo python experiments.py --model_name $1 --batch_size ${batch_size} --epochs 20 \
    --data_file ${data_file} --types_file ${types_file} --miss_file ${miss_file} \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file ${save_file} --train 1\
    #--true_miss_file ${true_miss_file}
}

test_model(){
    sudo python experiments.py --model_name $1 --batch_size 10000000 --epochs 1 \
    --data_file ${data_file} --types_file ${types_file} --miss_file ${miss_file} \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file ${save_file} --train 0 --restore 1 \
    #--true_miss_file ${true_miss_file}
}


declare save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}
train_model ${model} ${z_dim} ${y_dim} ${s_dim}
test_model ${model} ${z_dim} ${y_dim} ${s_dim}