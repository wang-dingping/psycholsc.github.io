DATAS SEGMENT
message	db 	 'input data please:',0dh,0ah,'$'
message2	db	'output:',0dh,0ah,'$'
maxlen	db	6								; 6
 		db	0								; 0
 		db	6 dup(9)						; 999999
display	db	 ' 0~ 9:',?,?,0dh,0ah
		db	 '10~99:',?,?,0dh,0ah
		db	 '100~ :',?,?
cr		db	0dh,0ah,'$'
count0	db	0
count1	db	0
count2	db	0

DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX								; Prepare
    
    MOV	CX,10								; CX=10. Counter
    LEA	DX,message							; DX=OFFSET Message
    MOV	AH,9								; AH=9.Output Message in DX.
    int	21h									; Output.
next:
	CALL read_exch                     		; CALL READ_EXCH, stored in DX.

	CMP	DX,10								; compare dx & 10
	JAE	g_10								; JAE (>=?JMP:continue)
	INC	count0       						; INC ++. 0~9
	JMP	a_num_end							; JMP to a_num_end
g_10:
	CMP	DX,100	 							; compare DX & 100      
	JAE	g_100								; JAE (>=?JMP:continue)
	INC	count1								; ++. 10~99
	JMP	a_num_end							; GOTO a_number_end
g_100:
	INC	count2								; ++. 100~
a_num_end:
	LOOP next								; loop for next 

	LEA	DX,message2							; DX=OFFSET message2, ready for output.
    MOV	AH,9								;
    INT	21H									; output.
	MOV	BX,offset display					; BX = offset display.
	MOV	AL,count0							; AL = count0
	CALL exch_to_asc10						; GOTO

	mov	[bx+6],ah
	mov 	[bx+7],al
	mov	al,count1
	call	exch_to_asc10
	mov	[bx+16],ah
	mov 	[bx+17],al
	mov	al,count2
	call	exch_to_asc10
	mov	[bx+26],ah
	mov	 [bx+27],al
	mov	dx,bx
	mov	AH,9
	int	21H


read_exch proc near
    PUSH CX									; push cx
	MOV	DX,offset maxlen					; DX = OFFSET maxlen
    MOV	AH,0AH								; ready for input, maxlen == [DX]
    INT	21H									; input.
    										; Attention for input: [DS:DX+1]==length.
    										; CL = maxlen+1 == real length of input. 

    LEA	DX,cr								; DX = OFFSET CR (\r\n)
	MOV	AH,9								; ready for output.
	INT	21H									; output a \r\n (CR == \r\n). 

    MOV	AX,0								; AX = 0
    MOV	CL,maxlen+1							; CL = maxlen + 1
    MOV BX,offset maxlen					; BX = offset maxlen
    ADD	BX,2								; BX += 2
again:										; loop head
	MOV	DH,0								; DH = 0
	MOV	DL,[BX]								; DL = * ptr BX
	AND	DL,0FH								; AND: lowest part remain.31H->01H, ascii to decimal
	MOV	CH,10								; CH = 10
	MUL	CH									; AX = AL*CH
	ADD	AX,DX								; AX+=DX
	INC	BX									; BX++
	DEC	CL									; CL--, counter--, until counter==0, pop CX and ret.
	JNZ	again								; JMP if NOT ZERO.

	MOV	DX,AX								; DX=AX
	POP	CX									; pop CX		
	RET										; Return
read_exch ENDP

exch_to_asc10 proc near
	MOV	AH,0
	MOV	DL,10
	DIV	DL
	OR	AX,3030H
	XCHG AH,AL
	RET
exch_to_asc10	endp
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START




