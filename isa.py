# Plugin for "POT OF GOLD" task
# Partially based on https://github.com/masthoon/BN_Loaders/blob/main/potluck_ctf.py
# And written for educational purposes

import binaryninja as bn


def u16(b: bytes):
    return int(b[::-1].hex(), 16)


def rname(idx):
    if idx <= 9:
        return f"R{idx}"
    assert False, "BUG"


def getSignedNumber(number, bitLength):
    mask = (2 ** bitLength) - 1
    if number & (1 << (bitLength - 1)):
        return number | ~mask
    else:
        return number & mask


class ISA:
    REG_SZ = 4
    INSTR_SZ = 4
    ADDR_SZ = 8
    ENTRYPOINT = 0x0
    HEADER = b'UNICORN\x00'

    OPCODE = {
        -1: 'UD2',
        0: 'NOP',
        1: 'JMP',
        2: 'LOAD',
        3: 'STORE',
        4: 'ALU',
        5: 'SYSCALL',
        6: 'CMP',
        7: 'MOV',
        8: 'PUSH',
        9: 'POP',
        10: 'CALL',
        11: 'UD1',
        12: 'RET',
    }

    JUMP_TYPES = {
        'GT': 'g',
        'LT': 'l',
        'E': 'eq',
        'NE': 'eq',
        'LE': 'le',
        'GE': 'ge'
    }


class ISASyscall(bn.CallingConvention):
    int_arg_regs = ['ID', 'R0', 'R1']
    int_return_reg = 'R0'
    eligible_for_heuristics = False


class ISArch(bn.Architecture):
    __memo = {}

    name = 'ISA'
    address_size = ISA.REG_SZ
    default_int_size = ISA.REG_SZ
    instr_alignment = ISA.INSTR_SZ
    max_instr_length = ISA.INSTR_SZ

    regs = {
        'R0': bn.RegisterInfo('R0', ISA.REG_SZ),
        'R1': bn.RegisterInfo('R1', ISA.REG_SZ),
        'R2': bn.RegisterInfo('R2', ISA.REG_SZ),
        'R3': bn.RegisterInfo('R3', ISA.REG_SZ),
        'R4': bn.RegisterInfo('R4', ISA.REG_SZ),
        'R5': bn.RegisterInfo('R5', ISA.REG_SZ),
        'R6': bn.RegisterInfo('R6', ISA.REG_SZ),
        'R7': bn.RegisterInfo('R7', ISA.REG_SZ),
        'R8': bn.RegisterInfo('R8', ISA.REG_SZ),
        'R9': bn.RegisterInfo('R9', ISA.REG_SZ),
        'ID': bn.RegisterInfo('ID', 4),
    }

    flags = ['eq', 'l', 'g', 'le', 'ge']

    link_reg = 'R9'
    stack_pointer = 'R8'

    def is_valid_instr(self, data: bytes):
        return len(data) >= 4 and data[0] <= 12 and data[0] in ISA.OPCODE

    def conv_val(self, addr, ba):
        if (addr & (1 << 15)) != 0:
            return (ba - ((1 << 16) - addr)), True
        return addr, False

    def get_instruction_low_level_il(
            self, data: bytes, addr: int, il: bn.LowLevelILFunction):
        wrapper = self.get_instruction_text(data, addr)

        if not wrapper:
            print('ABORTED')
            return None

        # Then we have a correct instruction text
        ops, _ = wrapper
        ops = self.__clean_op(ops)

        # And can reuse that for converting to LIL
        instruction = ops[0]
        operands = ops[1:] if len(ops) > 1 else []

        # For jump and call
        arg2_h = data[3]
        arg2_l = data[2]
        arg2 = arg2_l + (arg2_h << 8)
        sarg2 = getSignedNumber(arg2, 16)
        saddr = addr + sarg2

        match instruction['i']:
            case 'NOP':
                ni = il.nop()

            case 'LOAD':
                reg_dst = bn.RegisterName(bn.RegisterName(operands[0]['i']))
                reg_src = bn.RegisterName(bn.RegisterName(operands[1]['i']))

                v = il.load(data[1], il.reg(ISA.REG_SZ, reg_src))
                ni = il.set_reg(4, reg_dst, v)

            case 'STORE':
                reg_dst = bn.RegisterName(bn.RegisterName(operands[0]['i']))
                reg_src = bn.RegisterName(bn.RegisterName(operands[1]['i']))

                ni = il.store(
                    data[1], il.reg(
                        ISA.REG_SZ, reg_dst), il.reg(
                        ISA.REG_SZ, reg_src))

            case 'JMP' | 'JGT' | 'JLT' | 'JE' | 'JNE' | 'JLE' | 'JGE':

                # direct
                if instruction['i'] == 'JMP':
                    ni = il.append(
                        il.jump(
                            il.const_pointer(
                                ISA.REG_SZ,
                                saddr)))
                else:
                    j_type = instruction['i'][1:]
                    update_flag_bit = il.flag(ISA.JUMP_TYPES[j_type])

                    t = bn.LowLevelILLabel()
                    f = bn.LowLevelILLabel()

                    # check if it's jne - only neg instr
                    if instruction['i'] == 'JNE':
                        if_expr = il.if_expr(update_flag_bit, f, t)
                    else:
                        if_expr = il.if_expr(update_flag_bit, t, f)
                    il.append(if_expr)

                    # Mark labels
                    il.mark_label(t)
                    il.append(il.jump(il.const_pointer(ISA.REG_SZ, saddr)))

                    il.mark_label(f)

                    # Just for padding
                    ni = il.nop()

            case 'CMP':
                r1 = bn.RegisterName(operands[0]['i'])
                if operands[1]['t'] == bn.InstructionTextTokenType.RegisterToken:
                    r2 = bn.RegisterName(operands[1]['i'])

                    il.append(il.set_flag('le', il.compare_unsigned_less_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.reg(ISA.REG_SZ, r2)
                    )))

                    il.append(il.set_flag('ge', il.compare_unsigned_greater_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.reg(ISA.REG_SZ, r2)
                    )))

                    il.append(il.set_flag('g', il.compare_unsigned_greater_than(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.reg(ISA.REG_SZ, r2)
                    )))

                    il.append(il.set_flag('l', il.compare_unsigned_less_than(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.reg(ISA.REG_SZ, r2)
                    )))

                    ni = il.set_flag('eq', il.compare_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.reg(ISA.REG_SZ, r2)
                    ))

                else:
                    v = int(operands[1]['i'], 16)

                    il.append(il.set_flag('le', il.compare_unsigned_less_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.const(ISA.REG_SZ, v)
                    )))

                    il.append(il.set_flag('ge', il.compare_unsigned_greater_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.const(ISA.REG_SZ, v)
                    )))

                    il.append(il.set_flag('g', il.compare_unsigned_greater_than(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.const(ISA.REG_SZ, v)
                    )))

                    il.append(il.set_flag('l', il.compare_unsigned_less_than(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.const(ISA.REG_SZ, v)
                    )))

                    ni = il.set_flag('eq', il.compare_equal(
                        4, il.reg(ISA.REG_SZ, r1),
                        il.const(ISA.REG_SZ, v)
                    ))

            case 'PUSH':
                reg = bn.RegisterName(operands[0]['i'])
                ni = il.push(ISA.REG_SZ, il.reg(ISA.REG_SZ, reg))

            case 'POP':
                dest = bn.RegisterName(operands[0]['i'])
                val_from_stack = il.pop(ISA.REG_SZ)

                ni = il.set_reg(ISA.REG_SZ, dest, val_from_stack)

            case 'MOV':
                dest = bn.RegisterName(bn.RegisterName(operands[0]['i']))

                if operands[1]['t'] == bn.InstructionTextTokenType.RegisterToken:
                    src = il.reg(ISA.REG_SZ, bn.RegisterName(operands[1]['i']))
                else:
                    src = il.const(ISA.REG_SZ, int(operands[1]['i'], 16))

                ni = il.set_reg(4, dest, src)

            case 'SYSCALL':
                il.append(
                    il.set_reg(
                        4,
                        bn.RegisterName('ID'),
                        il.const(
                            4,
                            data[1])))
                ni = il.system_call()

            case 'CALL':
                if operands[0]['t'] == bn.InstructionTextTokenType.RegisterToken:
                    ni = il.call(
                        il.reg(
                            ISA.REG_SZ,
                            bn.RegisterName(
                                operands[0]['i'])))
                else:
                    ni = il.call(il.const_pointer(ISA.REG_SZ, saddr))

            case 'RET':
                ni = il.ret(il.reg(ISA.REG_SZ, bn.RegisterName(self.link_reg)))

            case 'ADD' | 'SUB' | 'MUL' | 'DIV' | 'AND' | 'OR' | 'XOR' | 'LSHIFT' | 'RSHIFT':
                dest = bn.RegisterName(operands[0]['i'])

                if operands[1]['t'] == bn.InstructionTextTokenType.RegisterToken:
                    v = il.reg(ISA.REG_SZ, bn.RegisterName(operands[1]['i']))
                else:
                    v = il.const(ISA.REG_SZ, int(operands[1]['i'], 16))

                match instruction['i']:
                    case 'ADD':
                        ni = il.set_reg(4, dest, il.add(4, il.reg(4, dest), v))
                    case 'SUB':
                        ni = il.set_reg(4, dest, il.sub(4, il.reg(4, dest), v))
                    case 'MUL':
                        ni = il.set_reg(
                            4, dest, il.mult(
                                4, il.reg(
                                    4, dest), v))
                    case 'DIV':
                        ni = il.set_reg(
                            4, dest, il.div_unsigned(
                                4, il.reg(
                                    4, dest), v))
                    case 'AND':
                        ni = il.set_reg(
                            4, dest, il.and_expr(
                                4, il.reg(
                                    4, dest), v))
                    case 'OR':
                        ni = il.set_reg(
                            4, dest, il.or_expr(
                                4, il.reg(
                                    4, dest), v))
                    case 'XOR':
                        ni = il.set_reg(
                            4, dest, il.xor_expr(
                                4, il.reg(
                                    4, dest), v))
                    case 'LSHIFT':
                        ni = il.set_reg(
                            4, dest, il.shift_left(
                                4, il.reg(
                                    4, dest), v))
                    case 'RSHIFT':
                        ni = il.set_reg(
                            4, dest, il.logical_shift_right(
                                4, il.reg(
                                    4, dest), v))
            case _:
                ni = il.unimplemented()

        il.append(ni)
        return ISA.INSTR_SZ

    def __clean_op(self, args):
        return [{'i': arg.text.replace(
            ',', '').strip(), 't': arg.type} for arg in args]

    def get_instruction_text(self, data: bytes, c_addr: int):
        if not self.is_valid_instr(data):
            print(hex(c_addr), 'invalid', data)
            return None

        opc = data[0]
        kind = data[1]
        args = [data[1], data[2], data[3]]
        addr = data[2] | (data[3] << 8)
        orig_addr = addr
        addr, _ = self.conv_val(addr, c_addr)

        ops = []
        match ISA.OPCODE[opc]:
            case 'NOP':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'NOP'
                    )
                )

            case 'LOAD':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'LOAD '
                    )
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[1]}, '
                    )
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[2]}'
                    )
                )

            case 'STORE':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'STORE '
                    )
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[2]}, '
                    )
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[1]}'
                    )
                )

            case 'ALU':
                some_v = bn.InstructionTextToken(
                    bn.InstructionTextTokenType.IntegerToken,
                    f'{hex(args[2])}'
                )

                if (args[0] & 0b1111) == 0:
                    some_v = bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[2]}'
                    )

                to_reg = bn.InstructionTextToken(
                    bn.InstructionTextTokenType.RegisterToken,
                    f'R{args[1]}, '
                )

                op = 'ALU'
                match args[0] >> 4:
                    case 0: op = 'ADD'
                    case 1: op = 'SUB'
                    case 2: op = 'MUL'
                    case 3: op = 'DIV'
                    case 4: op = 'AND'
                    case 5: op = 'OR'
                    case 6: op = 'XOR'
                    case 7: op = 'LSHIFT'
                    case 8: op = 'RSHIFT'

                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.TextToken,
                    f'{op} '
                ))

                ops.append(to_reg)
                ops.append(some_v)

            case 'JMP':
                match kind:
                    case 0:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JMP '
                        )
                    case 1:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JGT '
                        )
                    case 2:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JLT '
                        )
                    case 3:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JE '
                        )
                    case 4:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JLE '
                        )
                    case 5:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JGE '
                        )
                    case 6:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'JNE '
                        )

                    case _:
                        jt = bn.InstructionTextToken(
                            bn.InstructionTextTokenType.TextToken,
                            'UD1 '
                        )
                ops.append(jt)
                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.AddressDisplayToken,
                    f'{hex(addr)}'
                ))

                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.CommentToken,
                    f'{hex(orig_addr)}'
                ))

            case 'PUSH':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'PUSH '
                    ),
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{kind}'
                    )
                )

            case 'POP':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'POP '
                    )
                )
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{kind}'
                    )
                )

            case 'CALL':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        'CALL '
                    )
                )

                if kind == 0:
                    ops.append(
                        bn.InstructionTextToken(
                            bn.InstructionTextTokenType.AddressDisplayToken,
                            f'{hex(addr)}'
                        )
                    )
                elif kind == 1:
                    ops.append(
                        bn.InstructionTextToken(
                            bn.InstructionTextTokenType.RegisterToken,
                            f'R{args[1]}'
                        )
                    )

            case 'RET':
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        'RET'
                    )
                )

            case 'MOV':
                c_reg = args[0] >> 4
                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.TextToken,
                        f'MOV '
                    )
                )

                ops.append(
                    bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{c_reg}, '
                    )
                )

                if (args[0] & 0b1111) != 0:
                    ops.append(bn.InstructionTextToken(
                        bn.InstructionTextTokenType.IntegerToken,
                        f'{hex(addr)}'
                    ))
                else:
                    ops.append(bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[1]}'
                    ))

            case 'CMP':
                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.TextToken,
                    'CMP '
                ))

                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.RegisterToken,
                    f'R{args[1]},'
                ))

                v_type = (args[0] >> 4)
                if v_type == 1:
                    ops.append(bn.InstructionTextToken(
                        bn.InstructionTextTokenType.IntegerToken,
                        f'{hex(args[2])}'
                    ))

                elif v_type == 0:
                    ops.append(bn.InstructionTextToken(
                        bn.InstructionTextTokenType.RegisterToken,
                        f'R{args[2]}'
                    ))

            case 'SYSCALL':
                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.TextToken,
                    'SYSCALL '
                ))
                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.IntegerToken,
                    f'{hex(args[0])}'
                ))

            case _:
                ops.append(bn.InstructionTextToken(
                    bn.InstructionTextTokenType.TextToken,
                    'UD2'
                ))

        return ops, ISA.INSTR_SZ

    def get_instruction_info(self, data: bytes, addr: int):
        if not self.is_valid_instr(data):
            return None

        opc = data[0]
        kind = data[1]

        arg_addr = data[2] | (data[3] << 8)
        arg_addr, was_neg = self.conv_val(arg_addr, addr)
        if not was_neg:
            arg_addr += addr

        iif = bn.InstructionInfo(
            length=4
        )

        match ISA.OPCODE[opc]:
            case 'SYSCALL':
                # Kind - syscall number
                iif.add_branch(bn.BranchType.SystemCall, kind)

            case 'RET':
                iif.add_branch(bn.BranchType.FunctionReturn)

            case 'CALL':
                if kind == 1:
                    iif.add_branch(bn.BranchType.IndirectBranch)
                else:
                    iif.add_branch(bn.BranchType.CallDestination, arg_addr)
            case 'JMP':
                if kind == 0:
                    iif.add_branch(bn.BranchType.UnconditionalBranch, arg_addr)
                else:
                    iif.add_branch(bn.BranchType.TrueBranch, arg_addr)
                    iif.add_branch(bn.BranchType.FalseBranch, addr + 4)

        return iif


class ISASegment:
    def __init__(self, offset, raw):
        self.offset = offset

        self.addr = u16(raw[:2])
        self.size = u16(raw[2:4])
        self.prot = u16(raw[4:6])

    def register(self, bv: bn.BinaryView):

        print(
            f'Registering the {self.addr}, {self.size}, {self.addr}, {self.size}, {self.prot}')
        bv.add_auto_segment(
            self.addr,
            self.size,
            self.addr +
            self.offset,
            self.size,
            self.prot)


class ISAView(bn.BinaryView):
    name = 'ISA view'
    long_name = 'ISA ROM'

    def __init__(self, data: bn.BinaryView):
        bn.BinaryView.__init__(self, parent_view=data, file_metadata=data.file)

        self.data = data
        self.platform = bn.Architecture['ISA'].standalone_platform

        cc_ord = bn.CallingConvention(bn.Architecture['ISA'], 'ISACC')
        cc_ord.int_arg_regs = ['R0', 'R1', 'R2', 'R3']
        cc_ord.int_return_reg = 'R0'
        self.platform.convention = cc_ord

        # Register syscall convention
        cc_sys = ISASyscall(arch=bn.Architecture['ISA'], name='ISASyscall')
        bn.Architecture['ISA'].register_calling_convention(cc_ord)
        bn.Architecture['ISA'].standalone_platform.default_calling_convention = cc_ord
        bn.Architecture['ISA'].register_calling_convention(cc_sys)
        bn.Architecture['ISA'].system_call_convention = cc_sys
        self.platform.register_calling_convention(cc_sys)
        self.platform.system_call_convention = cc_sys

    def init(self):
        print('Initializing the ISAView')

        self.__parse_header()

        return True

    # Specifying arch options
    def perform_get_address_size(self) -> int:
        return ISA.ADDR_SZ

    def perform_get_entry_point(self):
        return ISA.ENTRYPOINT

    def perform_is_executable(self):
        return True

    def __parse_header(self):
        self.magic = self.data.read(0, 8)
        self.s_cnt = u16(self.data.read(8, 2))

        offset = 0xa
        base = 0xa + self.s_cnt * 6

        for i in range(self.s_cnt):
            raw = self.data.read(offset, 6)
            offset += 6

            ns = ISASegment(base, raw)
            ns.register(self)
            base += ns.size

    @classmethod
    def is_valid_for_data(self, data: bn.BinaryView):
        MAGIC = data.read(0, 8)
        print(MAGIC, ISA.HEADER)
        return MAGIC == ISA.HEADER
