#include <stdint.h>
#include "feature_module.h"
#include "def_nn1_nnvad.h"
#include "def_nn4_nnid.h"
#include "ambiq_nnsp_debug.h"
#include "nnsp_identification.h"
#include "nnid_class.h"
#include "PcmBufClass.h"
#include "nnidCntrlClass.h"

FeatureClass feat_vad, feat_nnid;
NNSPClass nnst_vad, nnst_nnid;
PcmBufClass pcmBuf_inst;
int16_t glob_th_prob = 0x7fff >> 1;
int16_t glob_count_trigger = 1;

void nnidCntrlClass_reset(nnidCntrlClass* pt_inst)
{
	PcmBufClass_reset(&pcmBuf_inst);
	NNSPClass_reset(&nnst_vad);
	NNSPClass_reset(&nnst_nnid);
	pt_inst->count_vad_trigger = 0;
	pt_inst->enroll_state = enroll_phase;
	pt_inst->acc_num_enroll = 0;
}
void nnidCntrlClass_init(nnidCntrlClass* pt_inst)
{
	
	pt_inst->pt_feat_nnid = (void*) &feat_nnid;
	pt_inst->pt_feat_vad = (void*) &feat_vad;
	pt_inst->pt_nnst_nnid = (void*)&nnst_nnid;
	pt_inst->pt_nnst_vad = (void*)&nnst_vad;
	pt_inst->pt_pcmBuf = (void*)&pcmBuf_inst;
	pt_inst->num_enroll = 4;
	// PCM_BUF init, reset

	PcmBufClass_init(&pcmBuf_inst);
	

	// nnvad init, reset
	NNSPClass_init(
		&nnst_vad,
		&net_nnvad,  // NeuralNetClass
		&feat_vad, // featureModule
		vad_id,
		feature_mean_nnvad,
		feature_stdR_nnvad,
		&glob_th_prob,
		&glob_count_trigger,
		&params_nn1_nnvad);
	

	// nnid init, reset
	NNSPClass_init(
		&nnst_nnid,
		&net_nnid,  // NeuralNetClass
		&feat_nnid, // featureModule
		nnid_id,
		feature_mean_nnid,
		feature_stdR_nnid,
		&glob_th_prob,
		&glob_count_trigger,
		&params_nn4_nnid);
}

int16_t nnidCntrlClass_exec(
			nnidCntrlClass* pt_inst,
			int16_t *rawPCM,
			float *pt_corr)
{
	int16_t detected;
	NNID_CLASS* pt_nnid;
	int16_t is_get_corr = 0;
	static int32_t embds[4 * 64];
	int32_t tmp32;
	pt_nnid = (NNID_CLASS*) nnst_nnid.pt_state_nnid;

	*pt_corr = pt_nnid->corr;
	
	
	PcmBufClass_setData(&pcmBuf_inst, rawPCM);
	detected = NNSPClass_exec(&nnst_vad, rawPCM);
	pt_inst->count_vad_trigger = (detected) ? pt_inst->count_vad_trigger + 1 : 0;
	
	if (pt_inst->count_vad_trigger == pt_nnid->thresh_get_corr)
	{
		is_get_corr = 1;
		for (int f = 0; f < pt_nnid->thresh_get_corr; f++)
		{
			PcmBufClass_getData(
				&pcmBuf_inst,
				pt_nnid->thresh_get_corr - f, // the lookback frame
				1, // frames to read
				rawPCM);
			if (f == pt_nnid->thresh_get_corr - 1)
				pt_nnid->is_get_corr = 1;
			else
				pt_nnid->is_get_corr = 0;
			NNSPClass_exec(&nnst_nnid, rawPCM);
		}
		if (pt_inst->enroll_state == enroll_phase)
		{
			NNSPClass_get_nn_out(embds + pt_inst->acc_num_enroll * pt_nnid->dim_embd,pt_nnid->dim_embd);
			pt_inst->acc_num_enroll+=1;
			if (pt_inst->acc_num_enroll == pt_inst->num_enroll)
			{
				for (int i = 0; i < pt_nnid->dim_embd; i++)
				{
					tmp32 = *(embds+i) + *(embds+i + pt_nnid->dim_embd) + *(embds+i + pt_nnid->dim_embd*2) + *(embds+i + pt_nnid->dim_embd*3);
					pt_nnid->pt_embd[i] = tmp32 >> 2;
				}
				pt_inst->enroll_state = test_phase;
			}
			*pt_corr = 0;
			pt_nnid->corr = -0.5;
		}
		else
			*pt_corr = pt_nnid->corr;
	

		NNSPClass_reset(&nnst_vad);
		NNSPClass_reset(&nnst_nnid);
		pt_inst->count_vad_trigger = 0;
	}
	return is_get_corr;
}