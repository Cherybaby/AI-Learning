/*
 * File: XPSbW_MotCtrlDMAPublicData.c
 *
 * Code generated for Simulink model 'MF015_DmaIf'.
 *
 * Model version                  : 1.10
 * Simulink Coder version         : 9.8 (R2022b) 13-May-2022
 * C/C++ source code generated on : Wed Jun 17 15:54:36 2026
 *
 * Target selection: autosar.tlc
 * Embedded hardware selection: Renesas->RH850
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "XPSbW_MotCtrlDMAPublicData.h"
#include "MF015_DmaIf.h"

/* Exported data definition */

/* Volatile memory section */
#define VCU_START_SEC_MONITOR
#include "Rte_MemMap.h"

/* Definition for custom storage class: MyMonitor */
MotCtrlIRQIfStruct MotCtrlIRQIf;
MotCtrlMngtAdc1IfStruct MotCtrlMngtAdc1If;
MotCtrlMngtAdc2IfStruct MotCtrlMngtAdc2If;
MotCtrlMngtIRQ2RteIfStruct MotCtrlMngtIRQ2RTEIf;
MotCtrlMngtIRQ2RteIfStruct MotCtrlMngtIRQ2RTEWriteIf;
MotCtrlMngtRTE2IRQIfStruct MotCtrlMngtRTE2IRQIf;
MotCtrlMngtRTE2IRQIfStruct MotCtrlMngtRTE2IRQReadIf;

#define VCU_STOP_SEC_MONITOR
#include "Rte_MemMap.h"

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
