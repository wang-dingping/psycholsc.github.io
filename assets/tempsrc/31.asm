DATAS SEGMENT
	A DW 1,2,3,4,5,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
	B DW 1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32  
	CC DW 20 dup(0)
DATAS ENDS

STACKS SEGMENT
    
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    mov DI,0
    mov SI,0
    mov CX,30
    mov BX,0
    mov DX,0
    lea SI,A
    lea DI,B
	
lo:
	mov AX,[SI]
	mov BX,[DI]
	cmp AX,BX
	JZ moveC	
con:	
	add DI,2
	loop lo
	
	add SI,2
	lea DI,B
	mov CX,30
	cmp SI,40
	JL lo
exit:
    MOV AH,4CH
    INT 21H
moveC:
	push DI
	push SI
	lea DI,CC
	add DI,DX
	CLD
	movsw
	pop SI
	pop DI
	add DI,2
	add DX,2
	jmp con
CODES ENDS
    END START
