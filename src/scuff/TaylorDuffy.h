/* Copyright (C) 2005-2011 M. T. Homer Reid
 *
 * This file is part of SCUFF-EM.
 *
 * SCUFF-EM is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * SCUFF-EM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/*
 * TaylorDuffy.h -- header file for implementation of the taylor-duffy
 *               -- scheme for computing singular panel-panel integrals
 *
 * homer reid    -- 5/2009 -- 2/2012
 */
#ifndef TAYLORDUFFY_H
#define TAYLORDUFFY_H

#include <complex>
typedef std::complex<double> cdouble;

/***************************************************************/
/* constants ***************************************************/
/***************************************************************/
//values for the WhichK parameter to the Taylor_xx routines 
#define TD_RP                   0
#define TD_HELMHOLTZ            1
#define TD_GRADHELMHOLTZ        2
#define TD_HIGHK_HELMHOLTZ      3
#define TD_HIGHK_GRADHELMHOLTZ  4
#define NUMKS                   5

//values for the WhichP parameter to the Taylor_xx routines
#define TD_UNITY               0
#define TD_RNORMAL             1
#define TD_PMCHWG1             2
#define TD_PMCHWC              3
#define TD_NMULLERG1           4
#define TD_NMULLERG2           5
#define TD_NMULLERC            6
#define NUMPS                  7

// values for the WhichCase parameter. note the values correspond to
// the numbers of common vertices in each case.
#define TD_COMMONVERTEX        1
#define TD_COMMONEDGE          2
#define TD_COMMONTRIANGLE      3

/*--------------------------------------------------------------*/
/*- TaylorDuffy() ---------------------------------------------*/
/*--------------------------------------------------------------*/
typedef struct TaylorDuffyArgStruct
 { 
    // mandatory input fields 
    int WhichCase;

    double *V1, *V2, *V3;
    double *V2P, *V3P;
 
    int NumPKs;
    int *PIndex;
    int *KIndex;
    cdouble *KParam;

    // output fields 
    cdouble *Result, *Error;
    int nCalls;

    // optional input fields 
    double *Q, *QP;
    double *nHat;

    double AbsTol, RelTol;
    int MaxEval;
    int ForceOnceIntegrable;

 } TaylorDuffyArgStruct;

void TaylorDuffy(TaylorDuffyArgStruct *Args);
void InitTaylorDuffyArgs(TaylorDuffyArgStruct *Args);

/***************************************************************/
/* relevant definitions for SGJ pcubature code, copied from    */
/* libSGJC.h in scuff-EM source distribution                   */
/***************************************************************/
typedef enum {
     ERROR_INDIVIDUAL = 0, /* individual relerr criteria in each component */
     ERROR_PAIRED, /* paired L2 norms of errors in each component,
		      mainly for integrating vectors of complex numbers */
     ERROR_L2, /* abserr is L_2 norm |e|, and relerr is |e|/|v| */
     ERROR_L1, /* abserr is L_1 norm |e|, and relerr is |e|/|v| */
     ERROR_LINF /* abserr is L_\infty norm |e|, and relerr is |e|/|v| */
} error_norm;

typedef int (*integrand) (unsigned ndim, const double *x, void *, 
                          unsigned fdim, double *fval);

extern "C" {
int pcubature(unsigned fdim, integrand f, void *fdata,
	      unsigned dim, const double *xmin, const double *xmax,
	      size_t maxEval, double reqAbsError, double reqRelError,
	      error_norm norm, double *val, double *err);
}

#endif
