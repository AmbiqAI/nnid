#ifndef __NNID_CNTRL_CLASS__
#define __NNID_CNTRL_CLASS__
#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
	enroll_phase 	= 0,
	test_phase		= 1,
}enroll_state_T;

typedef struct
{
	void *pt_feat_vad;
	void* pt_feat_nnid;
	void* pt_nnst_vad;
	void* pt_nnst_nnid;
	void* pt_pcmBuf;
	int16_t count_vad_trigger;
	enroll_state_T enroll_state;
	int8_t acc_num_enroll;
	int8_t num_enroll;
}nnidCntrlClass;


void nnidCntrlClass_reset(nnidCntrlClass* pt_inst);
void nnidCntrlClass_init(nnidCntrlClass *pt_inst);
int16_t nnidCntrlClass_exec(
	nnidCntrlClass* pt_inst,
	int16_t* rawPCM,
	float* pt_corr);
#ifdef __cplusplus
}
#endif
#endif