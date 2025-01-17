DATA		SEGMENT  PARA  'DATA'                   ; Define a data segment
		    ORG	200H                                ; Program starts from 0200H
													; See data by -d cs:200
B_BUF1		DB	'@',?                               ; Define Byte. '?' stands for undefined, program reserves a memory unit, initialized by 00H. IP += 1;
B_BUF2		DB	10,-4,0FCH,10,11001B,?,?,0          ; Define Byte.10=OAH;-4=FCH(0000 0100 -- 1111 1011 -- 1111 1100 -- FC Hex);0FCH=FCH(DB defines a BYTE);
													; 11001B= 11001 Binary=25 Decimal=19H;?=00H;0=00H
B_BUF3		DB	'COMPUTER'							; Every byte was changed into Hex code. 
B_BUF4		DB	01,'JAN',02,'FEB',03,'MAR'			; -
B_BUF5		DB	'1234'								; -
B_BUF6		DB	10 DUP(0)							; Duplicate 0 for 10 times.
B_BUF7      DB      5 DUP (2 DUP('*'),3 DUP(5))		; Duplicate * for 2 times, 05H for 3 times. Together duplicate for 5 times.
													; ============================
W_BUF1		DW	0FFFDH								; Define Word. 1 Word == 2 BYTE.FD is lower, 0FFFD = FD FF in memory.
W_BUF2		DW	-3									; -3:0000 0000 0000 0011 -- 1111 1111 1111 1100 -- 1111 1111 1111 1101 -- FF FD -- FD FF
W_BUF3		DW	11001B								; Same as Line 05, 25D = 19H
W_BUF4		DW	B_BUF4								; B_BUF4 Address=0212H, 12 02 in Memory. 
W_BUF5		DW	B_BUF7-B_BUF1						; 022C-0200=002C, 2C 00 in mem
W_BUF6		DW	1,2,3,4,5,3*20						; 00 01, 00 02, 00 03, 00 04, 00 05, 00 3C(60D=3CH). Reversed in mem.
W_BUF7      DW      5 DUP(0)						; 00 00 for 5 times.
W_BUF8		DW	$									; Address for now is 0265H. 65 02 in mem. 
													; ============================
D_BUF1		DD	?									; Define Double Words 00 00 00 00
D_BUF2		DD	'PC'								; 43H 50H 00H 00H-- C P
D_BUF3		DD	1234,12,34							; 1234 = 0000 0000 0000 0000 0000 0100 1101 0010 B = 00 00 04 D2 H = D2 04 00 00 in mem.
													; 12 = 0C 00 00 00 in mem. 34 = 22 00 00 00 in memory.
D_BUF4		DD	B_BUF4								; 0770:0212, 12 02 70 07 in memory.
D_BUF5      DD      W_BUF8-B_BUF7+1					; 3A 00 00 00
													; ============================
Q_BUF1		DQ	?									; Define Quad.00 00 00 00 00 00 00 00
Q_BUF2		DQ	3C32H								; 32 3C 00 00 00 00 00 00.
Q_BUF3		DQ	1234H								; 34 12 00 00 00 00 00 00.
													; ============================
T_BUF1		DT	?									; Define ten.00 00 00 00 00 00 00 00 00 00;
T_BUF2		DT	'PC'								; 43 50 00 00 00 00 00 00 00 00;
LEN		    DW	$									; Address:02AF, AF 02 in memory.
DATA		ENDS									; DATA ENDS.

STSEG       SEGMENT PARA STACK 'STACK'				; Stack
			DB	256	DUP('#')						; Duplicate '#' for 256 times in BYTE.
STSEG 		ENDS									; Stack Ends.
EDATA		SEGMENT PARA 'DATA'						; Extra DATA
			DB 	200H DUP('s')						; Duplicate 's' for 200H=512D times in BYTE.
EDATA		ENDS									; Extra Data ENDS.
CODE		SEGMENT PARA	'CODE'					; Code Segment
            ASSUME CS:CODE,DS:DATA,SS:STSEG,ES:EDATA; *1 See explanation under.
MAIN		PROC	FAR								; PROC and ENDP appear together; FAR see *2.
			MOV	AX,DATA								; 
			MOV	DS,AX								; These two lines gives a relationship between DS and DATA. Also in *1
			MOV	AX,EDATA							; 
			MOV	ES,AX								;
			LEA	BX,B_BUF1							; Give the EA of SRC to ORC.
            MOV DI,30h								;
AGAIN:		MOV	AL,[BX]								; A Loop Segment.
			MOV	ES:[DI],AL							;
			INC	BX									; +=1
			INC	DI									;
			CMP	BX,LEN								; Compare.
			JB	AGAIN								; CF==0 && ZF==0 (A<B)
			MOV	AX,4C00H							; Exit(1)
			INT	21H									; Exit(2)
MAIN		ENDP									; END PROC
CODE		ENDS									; END SEGMENT
			END	MAIN								; END MAIN






