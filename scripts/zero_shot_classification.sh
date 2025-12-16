FILENAME='NLM- Malaria Data.json' # json file of the corresponding data, stored in OmniMedVQA/QA_information/Open-access
OUTPUTNAME=NLM-Malaria # name of result folder
WORKDIR=$(pwd) # where OmniMedVQA folder is extracted
MODEL_CKPT=checkpoint_llava_med_instruct_60k_inline_mention_version_1-5_1e0_multi_graph_100_scale_dci_test_bugfix # path to folder checkpoint
CONNECTOR_TYPE=dci # use 'dci' if model trained with DCI, else 'none'


export PYTHONPATH=$PYTHONPATH:./exgra_med
python exgra_med/llava/eval/model_omnimed_vqa_eval.py \
--question-file "$WORKDIR/OmniMedVQA/QA_information/Open-access/$FILENAME" \
--model-name $MODEL_CKPT \
--mm_dense_connector_type $CONNECTOR_TYPE \
--num_l 6 \
--image-folder "$WORKDIR/OmniMedVQA" \
--answers_base_path OmnimedVQA_results/$OUTPUTNAME