/*
 * File: XPSbW_MotCtrlDMAPublicDataType.h
 *
 * Code generated for Simulink model 'MF015_DmaIf'.
 *
 * Model version                  : 1.10
 * Simulink Coder version         : 9.8 (R2022b) 13-May-2022
 * C/C++ source code generated on : Wed Jun 17 15:54:36 2026
 */

#ifndef RTW_HEADER_XPSbW_MotCtrlDMAPublicDataType_h_
#define RTW_HEADER_XPSbW_MotCtrlDMAPublicDataType_h_
#include "rtwtypes.h"

typedef struct {
  float32 ES013_f32_MotAnAgPSin_UL;
  float32 ES013_f32_MotAnAgPCos_UL;
  float32 ES013_f32_MotAnAgNSin_UL;
  float32 ES013_f32_MotAnAgNCos_UL;
  float32 ES014_f32_MotAnAg_deg;
  float32 ES003_f32_MotAgArbd_deg;
  float32 ES005_f32_MotCurrAMeasArbt_A;
  float32 ES005_f32_MotCurrBMeasArbt_A;
  float32 ES005_f32_MotCurrCMeasArbt_A;
  boolean ES005_b_CurrSenseErrorFlag_flg;
  float32 ES006_f32_MotCurrDAxis_A;
  float32 ES006_f32_MotCurrQAxis_A;
  float32 MF006_f32_MotCtrlVltgDAxisCmd_V;
  float32 MF006_f32_MotCtrlVltgQAxisCmd_V;
  float32 MF007_f32_MotCtrPwmAPha_UL;
  float32 MF007_f32_MotCtrPwmBPha_UL;
  float32 MF007_f32_MotCtrPwmCPha_UL;
  uint8 MF007_u8_MotAgCalibrationSt_st_gdu8;
  float32 MF007_f32_PreMotZeroPosOffset_deg_gdf32;
  uint8 ES014_u8_MotAgChannel_UL;
} MotCtrlIRQIfStruct;

typedef struct {
  float32 ES004_f32_MotVelElecFild_radps;
  float32 ES008_f32_MotBrdgVltg_V;
  float32 MF001_f32_MotKeEstimn_Vpradps;
  float32 MF001_f32_MotQAxisInduEstimn_Henry;
  float32 MF001_f32_MotDAxisInduEstimn_Henry;
  float32 MF001_f32_MotREstimn_Ohm;
  float32 MF002_f32_MotCurrLoopDAxisKi_UL;
  float32 MF002_f32_MotCurrLoopDAxisKp_UL;
  float32 MF002_f32_MotCurrLoopQAxisKi_UL;
  float32 MF002_f32_MotCurrLoopQAxisKp_UL;
  float32 MF014_f32_MotVFCtrlVltgAlpha_V;
  float32 MF014_f32_MotVFCtrlVltgBeta_V;
  uint8 MF014_u8_MotCtrlMode_UL;
  float32 ES008_f32_MotCurrVROMeas_V;
  float32 MF005_f32_MotCurrDAxisCmd_A;
  float32 MF005_f32_MotCurrQAxisCmd_A;
} MotCtrlMngtRTE2IRQIfStruct;

typedef struct {
  float32 PhyValue_MotCurrAMeas_mV;
  float32 PhyValue_MotCurrBMeas_mV;
  float32 PhyValue_MotCurrCMeas_mV;
  float32 PhyValue_MotAnAgSinNRaw_mV;
  float32 PhyValue_MotAnAgSinPRaw_mV;
  float32 PhyValue_MotAnAgCosNRaw_mV;
  float32 PhyValue_MotAnAgCosPRaw_mV;
} MotCtrlMngtAdc2IfStruct;

typedef struct {
  float32 ES003_f32_MotAgArbd_deg;
  float32 ES006_f32_MotCurrDAxis_A;
  float32 ES006_f32_MotCurrQAxis_A;
} MotCtrlMngtIRQ2RteIfStruct;

typedef struct {
  float32 ControllerKL30Vltg;
  float32 MotNtcTemp1Mos1;
  float32 MotNtcTemp1Mos2;
  float32 MotNtcTemp1Mos3;
} MotCtrlMngtAdc1IfStruct;

#endif                        /* RTW_HEADER_XPSbW_MotCtrlDMAPublicDataType_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
