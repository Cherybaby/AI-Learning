/*
 * File: XPSbW_MotCtrlDMAPublicData.h
 *
 * Code generated for Simulink model 'MF015_DmaIf'.
 *
 * Model version                  : 1.4
 * Simulink Coder version         : 9.8 (R2022b) 13-May-2022
 * C/C++ source code generated on : Wed Jun 17 15:08:46 2026
 */

#ifndef RTW_HEADER_XPSbW_MotCtrlDMAPublicData_h_
#define RTW_HEADER_XPSbW_MotCtrlDMAPublicData_h_
#include "XPSbW_MotCtrlDMAPublicDataType.h"

/* Volatile memory section */
#define VCU_START_SEC_MONITOR
#include "Rte_MemMap.h"

/* Exported data declaration */
/* Declaration for custom storage class: MyMonitor */
extern MotCtrlIRQIfStruct MotCtrlIRQIf;
extern MotCtrlMngtAdc1IfStruct MotCtrlMngtAdc1If;
extern MotCtrlMngtAdc2IfStruct MotCtrlMngtAdc2If;
extern MotCtrlMngtIRQ2RteIfStruct MotCtrlMngtIRQ2RTEIf;
extern MotCtrlMngtIRQ2RteIfStruct MotCtrlMngtIRQ2RTEWriteIf;
extern MotCtrlMngtRTE2IRQIfStruct MotCtrlMngtRTE2IRQIf;
extern MotCtrlMngtRTE2IRQIfStruct MotCtrlMngtRTE2IRQReadIf;

#define VCU_STOP_SEC_MONITOR
#include "Rte_MemMap.h"
#endif                            /* RTW_HEADER_XPSbW_MotCtrlDMAPublicData_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
