# Run experiments of EntLM on rfc
#ID=-1
ID=1

for p in {"TCP","DCCP","SCTP","PPTP","LTP","BGPv4"}
#for p in {"SCTP","PPTP","LTP","DCCP","BGPv4"}
do

PROTOCOL=${p}
MAP_PATH=data/label_map_handmade.json
#ID=$[(${ID}+1)%4]

echo ""
echo "---------------------------------------Training with file ${PROTOCOL}----------------------------------------"
echo ""
nohup python3 models/entlm_crf_rfc.py                        			\
			--features    \
			--savedir .   \
			--do_train    \
			--do_eval     \
			--protocol ${PROTOCOL}   \
			--outdir output 	\
			--bert_model networking_bert_rfcs_only    \
			--learning_rate 1e-5  \
			--batch_size 1 	      \
			--patience 10         \
			--cuda_device ${ID}       \
			>log_entlm_crf_ds_lr1e-5_${PROTOCOL}.txt 2>&1
done
python3 scripts/read_results.py --logs_name log_entlm_crf_ds_lr1e-5