#include "Int.h"
#include "Random.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <emmintrin.h>

// ------------------------------------------------

#define MAX(x,y) (((x)>(y)) ? (x):(y))
#define MIN(x,y) (((x)<(y)) ? (x):(y))

#define NB64BLOCK 5
#define NB32BLOCK 10

#define BASELENGTH 32

static uint32_t isqrt(uint64_t x) {

  uint64_t q = 1;
  uint64_t r = 0;

  while(q <= x) {
    q <<= 2;
  }
  q >>= 2;

  while(q != 0) {
    uint64_t t = r;
    r = (r + q) >> 1;
    if(x < r * r) {
      r = t;
    }
    q >>= 1;
  }

  return (uint32_t)r;

}

// ------------------------------------------------

Int::Int(Int *a) {
  if(a) Set(a);
  else SetInt32(0);
}

// ------------------------------------------------

Int::Int(int64_t i64) {

  if (i64<0) {
    // negative
    CLEAR();
    bits64[0] = -i64;
    SetNegative(true);
  } else {
    CLEAR();
    bits64[0] = i64;
  }

}

// ------------------------------------------------

Int::Int(uint64_t u64) {

  CLEAR();
  bits64[0] = u64;

}

// ------------------------------------------------

void Int::CLEAR() {
  memset(bits64,0, NB64BLOCK * 8);
}

// ------------------------------------------------

void Int::Clear() {
  CLEAR();
}

// ------------------------------------------------

void Int::SetInt64(int64_t s64) {

  if (s64<0) {
    SetInt64(-s64);
    SetNegative(true);
  } else {
    SetInt64((uint64_t)s64);
    SetNegative(false);
  }

}

// ------------------------------------------------

void Int::SetInt64(uint64_t u64) {
  CLEAR();
  bits64[0] = u64;
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {

  CLEAR();
  bits[0] = value;

}

// ------------------------------------------------

void Int::Set(Int *a) {

  negative = a->negative;
  memcpy(bits64,a->bits64,sizeof(bits64));

}

// ------------------------------------------------

void Int::Add(uint64_t a) {

  int i = 0;
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[i], a, bits64 + i);
  i++;
  while(c!=0 && i<NB64BLOCK)
    c = _addcarry_u64(c, bits64[i], 0, bits64 + i++);

  if( (bits64[NB64BLOCK-1] & 0x8000000000000000ULL) != 0 )
    SetNegative(true);

}

// ------------------------------------------------

void Int::Add(Int *a) {

  if(negative == a->negative) {

    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);

    if( (bits64[NB64BLOCK-1] & 0x8000000000000000ULL) != 0 )
      SetNegative(true);

  } else {

    if(negative) {
      Set(a);
      negative = false;
      Sub(this);
    } else {
      Sub(a);
    }

  }

}

// ------------------------------------------------

void Int::Add(Int *a,Int *b) {

  unsigned char c = 0;
  c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);

  if( (bits64[NB64BLOCK-1] & 0x8000000000000000ULL) != 0 )
    SetNegative(true);
  else
    SetNegative(false);

}

// ------------------------------------------------

void Int::Sub(uint64_t a) {

  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  int i = 1;
  while(c!=0 && i<NB64BLOCK)
    c = _subborrow_u64(c, bits64[i], 0, bits64 + i++);

  if( (bits64[NB64BLOCK-1] & 0x8000000000000000ULL) != 0 )
    SetNegative(true);

}

// ------------------------------------------------

void Int::Sub(Int *a) {

  if(negative == a->negative) {

    unsigned char c = 0;
    c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 + 4);

    if( (bits64[NB64BLOCK-1] & 0x8000000000000000ULL) != 0 )
      SetNegative(true);

  } else {

    if(negative) {
      Add(a);
      SetNegative(true);
    } else {
      Add(a);
      SetNegative(false);
    }

  }

}

// ------------------------------------------------

bool Int::IsPositive() {
  return !negative;
}

// ------------------------------------------------

bool Int::IsNegative() {
  return negative;
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {

  Int b(a);
  b.Sub(this);
  return b.IsPositive();

}

// ------------------------------------------------

bool Int::IsLower(Int *a) {

  Int b(a);
  b.Sub(this);
  return b.IsNegative();

}

// ------------------------------------------------

bool Int::IsEqual(Int *a) {

  return
  (bits64[0] == a->bits64[0]) &&
  (bits64[1] == a->bits64[1]) &&
  (bits64[2] == a->bits64[2]) &&
  (bits64[3] == a->bits64[3]) &&
  (bits64[4] == a->bits64[4]);

}

// ------------------------------------------------

bool Int::IsZero() {

  return (bits64[0]|bits64[1]|bits64[2]|bits64[3]|bits64[4])==0;

}

// ------------------------------------------------

void Int::SetNull() {
  CLEAR();
  negative = false;
}

// ------------------------------------------------

bool Int::IsNull() {
  return IsZero() && !negative;
}

// ------------------------------------------------

bool Int::IsOne() {
  return (bits64[0]==1) && (bits64[1]==0) && (bits64[2]==0) && (bits64[3]==0) && (bits64[4]==0);
}

// ------------------------------------------------

bool Int::IsEven() {
  return (GetInt32() & 1)==0;
}

// ------------------------------------------------

bool Int::IsOdd() {
  return (GetInt32() & 1)==1;
}

// ------------------------------------------------

uint32_t Int::GetInt32() {
  return bits[0];
}

// ------------------------------------------------

int Int::GetBitLength() {

  int i = NB32BLOCK - 1;
  uint32_t mask = 0x80000000;

  while(i >= 0 && bits[i] == 0)
    i--;

  if (i < 0)
    return 0;

  while(mask != 0 && (bits[i] & mask) == 0)
    mask >>= 1;

  return i * 32 + 32 - __builtin_clz(mask);

}

// ------------------------------------------------

int Int::GetSize() {

  int i = NB64BLOCK - 1;
  while(i > 0 && bits64[i] == 0) i--;
  return i + 1;

}

// ------------------------------------------------

int Int::GetSize32() {

  int i = NB32BLOCK - 1;
  while(i > 0 && bits[i] == 0) i--;
  return i + 1;

}

// ------------------------------------------------

void Int::Mult(Int *a) {

  Int b(this);
  Mult(a,&b);

}

// ------------------------------------------------

void Int::Mult(uint64_t a) {

  unsigned char c = 0;
  uint64_t h;
  for (int i = 0; i < NB64BLOCK; i++) {
    __int128 r = (__int128)bits64[i] * a + c;
    bits64[i] = r;
    c = r >> 64;
  }

}

// ------------------------------------------------

void Int::IMult(int64_t a) {

  bool sn = negative;
  bool an = a<0;
  if(an) {
    negative = !negative;
    a = -a;
  }

  unsigned char c = 0;
  uint64_t h;
  for (int i = 0; i < NB64BLOCK; i++) {
    __int128 r = (__int128)bits64[i] * a + c;
    bits64[i] = r;
    c = r >> 64;
  }

  if(sn!=an && !IsZero())
    negative = !negative;

}

// ------------------------------------------------

void Int::Mult(Int *a, uint64_t b) {

  unsigned char c = 0;
  uint64_t h;
  for (int i = 0; i < NB64BLOCK; i++) {
    __int128 r = (__int128)a->bits64[i] * b + c;
    bits64[i] = r;
    c = r >> 64;
  }

}

// ------------------------------------------------

void Int::Mult(Int *a, Int *b) {

  if(a->IsZero() || b->IsZero()) {
    CLEAR();
    return;
  }

  int s = b->GetSize();

  CLEAR();

  for (int i = 0; i < s; i++) {

    unsigned char c = 0;
    uint64_t h;
    for (int j = 0; j < NB64BLOCK; j++) {
      int k = i + j;
      if (k >= NB64BLOCK) break;
      __int128 r = (__int128)a->bits64[j] * b->bits64[i] + bits64[k] + c;
      bits64[k] = r;
      c = r >> 64;
    }

  }

  negative = a->negative != b->negative;

}

// ------------------------------------------------

void Int::Div(Int *a, Int *mod) {

  if (a->IsZero()) {
    if(mod) mod->Set(this);
    SetInt32(0);
    return;
  }

  if (IsZero()) {
    if(mod) mod->SetInt32(0);
    return;
  }

  if (a->IsGreater(this)) {
    if(mod) mod->Set(this);
    SetInt32(0);
    return;
  }

  Int _p(this);
  _p.negative = false;
  a->negative = false;
  negative = false;

  CLEAR();
  Int rem(& _p);
  Int current(a);
  Int temp;

  int aSize = a->GetSize();
  int dSize = _p.GetSize() - aSize;

  current.ShiftL(dSize * 64);

  for (int i = dSize; i >= 0; i--) {

    int n = 64;
    uint64_t qBase = current.bits64[i + aSize - 1];
    if (qBase < a->bits64[aSize - 1]) n--;

    for (int j = n - 1; j >= 0; j--) {

      current.ShiftL(1);
      if (current.IsGreaterOrEqual(a)) {
        current.Sub(a);
        bits64[i] += (uint64_t)1 << j;
      }

    }

  }

  if(mod) mod->Set(&current);

  a->negative = false;
  negative = _p.negative != a->negative;

}

// ------------------------------------------------

void Int::GCD(Int *a) {

  Int *u = this;
  Int v(a);
  Int r;

  while (!v.IsZero()) {
    u->Div(&v,&r);
    u = &v;
    v.Set(&r);
  }

  Set(u);

}

// ------------------------------------------------

void Int::SetBase10(const char *value) {

  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);
  bool neg = false;

  if(value[0]=='-') {
    neg = true;
    value++;
    lgth--;
  }

  for (int i = lgth - 1; i >= 0; i--) {
    c.SetInt32(value[i] - '0');
    c.Mult(&pw);
    Add(&c);
    pw.Mult(10);
  }

  if(neg)
    negative = !IsZero();

}

// ------------------------------------------------

void Int::SetBase16(const char *value) {

  SetInt32(0);
  Int pw((uint64_t)1);
  Int c;

  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    int cv = value[i];
    if (cv >= '0' && cv <= '9')
      c.SetInt32(cv - '0');
    else if (cv >= 'A' && cv <= 'F')
      c.SetInt32(cv - 'A' + 10);
    else if (cv >= 'a' && cv <= 'f')
      c.SetInt32(cv - 'a' + 10);
    else
      c.SetInt32(0);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(16);
  }

}

// ------------------------------------------------

const char *Int::GetBase10() {

  return GetCStr(10);

}

// ------------------------------------------------

char* Int::GetCStr(int base) {

  static char s[1024];
  char *ret = s;
  int idx = 0;

  if (base != 10) {
    // Not implemented
    strcpy(s,"?");
    return s;
  }

  Int v(this);
  v.negative = false;
  if (v.IsZero()) {
    strcpy(s,"0");
    return s;
  }

  while (!v.IsZero()) {

    Int remainder;
    v.Div(base, &remainder);
    s[idx] = remainder.GetInt32() + '0';
    idx++;

  }

  if (negative) {
    s[idx] = '-';
    idx++;
  }

  s[idx] = 0;

  // reverse
  int len = (int)strlen(s);
  char swap;
  for (int i = 0; i < len / 2; i++) {
    swap = s[i];
    s[i] = s[len - 1 - i];
    s[len - 1 - i] = swap;
  }

  return ret;

}

// ------------------------------------------------

const char *Int::GetBase16() {

  return GetCStr(16);

}

// ------------------------------------------------

void Int::Set32Bytes(const unsigned char *bytes) {

  CLEAR();
  bits64[3] = ((uint64_t)bytes[0] << 56) |
              ((uint64_t)bytes[1] << 48) |
              ((uint64_t)bytes[2] << 40) |
              ((uint64_t)bytes[3] << 32) |
              ((uint64_t)bytes[4] << 24) |
              ((uint64_t)bytes[5] << 16) |
              ((uint64_t)bytes[6] << 8) |
              ((uint64_t)bytes[7]);
  bits64[2] = ((uint64_t)bytes[8] << 56) |
              ((uint64_t)bytes[9] << 48) |
              ((uint64_t)bytes[10] << 40) |
              ((uint64_t)bytes[11] << 32) |
              ((uint64_t)bytes[12] << 24) |
              ((uint64_t)bytes[13] << 16) |
              ((uint64_t)bytes[14] << 8) |
              ((uint64_t)bytes[15]);
  bits64[1] = ((uint64_t)bytes[16] << 56) |
              ((uint64_t)bytes[17] << 48) |
              ((uint64_t)bytes[18] << 40) |
              ((uint64_t)bytes[19] << 32) |
              ((uint64_t)bytes[20] << 24) |
              ((uint64_t)bytes[21] << 16) |
              ((uint64_t)bytes[22] << 8) |
              ((uint64_t)bytes[23]);
  bits64[0] = ((uint64_t)bytes[24] << 56) |
              ((uint64_t)bytes[25] << 48) |
              ((uint64_t)bytes[26] << 40) |
              ((uint64_t)bytes[27] << 32) |
              ((uint64_t)bytes[28] << 24) |
              ((uint64_t)bytes[29] << 16) |
              ((uint64_t)bytes[30] << 8) |
              ((uint64_t)bytes[31]);

}

// ------------------------------------------------

void Int::Get32Bytes(unsigned char *buff) {

  uint64_t val;

  val = bits64[3];
  buff[0] = (unsigned char)(val >> 56);
  buff[1] = (unsigned char)(val >> 48);
  buff[2] = (unsigned char)(val >> 40);
  buff[3] = (unsigned char)(val >> 32);
  buff[4] = (unsigned char)(val >> 24);
  buff[5] = (unsigned char)(val >> 16);
  buff[6] = (unsigned char)(val >> 8);
  buff[7] = (unsigned char)(val);

  val = bits64[2];
  buff[8] = (unsigned char)(val >> 56);
  buff[9] = (unsigned char)(val >> 48);
  buff[10] = (unsigned char)(val >> 40);
  buff[11] = (unsigned char)(val >> 32);
  buff[12] = (unsigned char)(val >> 24);
  buff[13] = (unsigned char)(val >> 16);
  buff[14] = (unsigned char)(val >> 8);
  buff[15] = (unsigned char)(val);

  val = bits64[1];
  buff[16] = (unsigned char)(val >> 56);
  buff[17] = (unsigned char)(val >> 48);
  buff[18] = (unsigned char)(val >> 40);
  buff[19] = (unsigned char)(val >> 32);
  buff[20] = (unsigned char)(val >> 24);
  buff[21] = (unsigned char)(val >> 16);
  buff[22] = (unsigned char)(val >> 8);
  buff[23] = (unsigned char)(val);

  val = bits64[0];
  buff[24] = (unsigned char)(val >> 56);
  buff[25] = (unsigned char)(val >> 48);
  buff[26] = (unsigned char)(val >> 40);
  buff[27] = (unsigned char)(val >> 32);
  buff[28] = (unsigned char)(val >> 24);
  buff[29] = (unsigned char)(val >> 16);
  buff[30] = (unsigned char)(val >> 8);
  buff[31] = (unsigned char)(val);

}

// ------------------------------------------------

void Int::SetByte(int byteNum, unsigned char byte) {

  ( (unsigned char *)bits )[byteNum] = byte;

}

// ------------------------------------------------

unsigned char Int::GetByte(int byteNum) {

  return ( (unsigned char *)bits )[byteNum];

}

// ------------------------------------------------

void Int::SetDWord(int wordNum, uint32_t val) {

  bits[wordNum] = val;

}

// ------------------------------------------------

void Int::SetQWord(int wordNum, uint64_t val) {

  bits64[wordNum] = val;

}

// ------------------------------------------------

void Int::Rand(int nbits) {

  int nb = (nbits + 7) / 8;
  unsigned char bytes[64];
  rndl(bytes, nb);
  Set32Bytes(bytes);
  bits[NB32BLOCK - 1] &= (0xFFFFFFFF >> (32 - (nbits % 32)));

}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {

  Int s;
  int d = n / 64;
  int rem = n % 64;

  for (int i = NB64BLOCK - 1; i >= d; i--) {
    if (i - d >= 0)
      bits64[i] = (bits64[i - d] << rem);
    if (i - d - 1 >= 0)
      bits64[i] |= (bits64[i - d - 1] >> (64 - rem));
  }
  for (int i = d - 1; i >= 0; i--) {
    bits64[i] = 0;
  }

}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {

  Int s;
  int d = n / 64;
  int rem = n % 64;

  for (int i = 0; i < NB64BLOCK; i++) {
    if (i + d < NB64BLOCK)
      bits64[i] = (bits64[i + d] >> rem);
    if (i + d + 1 < NB64BLOCK)
      bits64[i] |= (bits64[i + d + 1] << (64 - rem));
  }

}

// ------------------------------------------------

void Int::AddOne() {

  unsigned char c = 1;
  c = _addcarry_u64(c, bits64[0], 1ULL, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0ULL, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0ULL, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0ULL, bits64 + 4);

}

// ------------------------------------------------

void Int::SubOne() {

  unsigned char c = 1;
  c = _subborrow_u64(c, bits64[0], 1ULL, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0ULL, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0ULL, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0ULL, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0ULL, bits64 + 4);

}

// ------------------------------------------------

int Int::GetBit(uint32_t n) {

  uint32_t byte = n >> 5;
  uint32_t bit = n & 31;
  uint32_t mask = 1 << bit;
  return (bits[byte] & mask) >> bit;

}

// ------------------------------------------------

void Int::SetBit(uint32_t n, int val) {

  uint32_t byte = n >> 5;
  uint32_t bit = n & 31;
  uint32_t mask = 1 << bit;
  if (val)
    bits[byte] |= mask;
  else
    bits[byte] &= (0xFFFFFFFF - mask);

}

// ------------------------------------------------

void Int::Mod(Int *p) {

  if (IsGreater(p)) {
    Sub(p);
  }

  while (IsGreaterOrEqual(p)) {
    Sub(p);
  }

}

// ------------------------------------------------

void Int::ModAdd(Int *a, Int *mod) {

  Add(a);
  Mod(mod);

}

// ------------------------------------------------

void Int::ModSub(Int *a, Int *mod) {

  Sub(a);
  if (IsNegative()) Add(mod);

}

// ------------------------------------------------

void Int::ModMult(Int *a, Int *b, Int *mod) {

  Int t;
  t.Mult(a, b);
  t.Mod(mod);
  Set(&t);

}

// ------------------------------------------------

void Int::ModExp(Int *e, Int *mod) {

  Int base(this);
  SetInt32(1);

  int i = 0;
  uint32_t nbBit = e->GetBitLength();
  for (int i = nbBit - 1; i >= 0; i--) {

    Int tmp;
    tmp.ModMult(this, this, mod);

    if (e->GetBit(i))
      Set(&tmp.ModMult(&tmp, &base, mod));
    else
      Set(&tmp);

  }

}

// ------------------------------------------------

void Int::ModInv(Int *mod) {

  Int u1(1);
  Int u3(this);
  Int v1(0);
  Int v3(mod);
  Int t1;
  Int t3;
  Int q;

  while (!v3.IsZero()) {

    q.Div(&u3, &t3);
    t1.Set(&u1);
    t1.ModSub(&q.ModMult(&q, &v1, mod), mod);

    u1.Set(&v1);
    u3.Set(&v3);
    v1.Set(&t1);
    v3.Set(&t3);

  }

  Set(&u1);

}

// ------------------------------------------------

void Int::ModNeg(Int *mod) {

  Sub(this, mod);
  negative = !negative;

}

// ------------------------------------------------

void Int::Neg() {

  negative = !negative && !IsZero();

}

// ------------------------------------------------

void Int::SwapByteOrder() {

  for (int i = 0; i < NB32BLOCK / 2; i++) {
    uint32_t swap = bits[i];
    bits[i] = bits[NB32BLOCK - 1 - i];
    bits[NB32BLOCK - 1 - i] = swap;
  }

}

// ------------------------------------------------

Int *Int::GetFieldCharacteristic() {

  static Int _p;
  if (_p.IsZero()) {
    _p.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  }
  return &_p;

}

// ------------------------------------------------

Int *Int::GetR() {

  static Int _r;
  if (_r.IsZero()) {
    _r.SetBase16("10000000000000000000000000000000000000000000000000000000000000000");
  }
  return &_r;

}

// ------------------------------------------------

Int *Int::GetR2(Int *p) {

  static Int _r2;
  if (_r2.IsZero()) {
    Int t1 = *GetR();
    _r2.ModMult(&t1, &t1, p);
  }
  return &_r2;

}

// ------------------------------------------------

Int *Int::GetR3(Int *p) {

  static Int _r3;
  if (_r3.IsZero()) {
    Int t1 = *GetR();
    Int t2 = *GetR2(p);
    _r3.ModMult(&t1, &t2, p);
  }
  return &_r3;

}

// ------------------------------------------------

Int *Int::GetR4(Int *p) {

  static Int _r4;
  if (_r4.IsZero()) {
    Int t1 = *GetR2(p);
    _r4.ModMult(&t1, &t1, p);
  }
  return &_r4;

}

// ------------------------------------------------

void Int::REDUCE(uint64_t *t, Int *p) {

  uint64_t c;
  uint64_t *pv = p->bits64;

  c = _subborrow_u64(0, t[0], pv[0], t + 0);
  c = _subborrow_u64(c, t[1], pv[1], t + 1);
  c = _subborrow_u64(c, t[2], pv[2], t + 2);
  c = _subborrow_u64(c, t[3], pv[3], t + 3);
  c = _subborrow_u64(c, t[4], pv[4], t + 4);
  c = _subborrow_u64(c, t[5], 0, t + 5);
  c = _subborrow_u64(c, t[6], 0, t + 6);
  c = _subborrow_u64(c, t[7], 0, t + 7);
  c = _subborrow_u64(c, t[8], 0, t + 8);
  c = _subborrow_u64(c, t[9], 0, t + 9);

  if (c) {
    c = _addcarry_u64(0, t[0], pv[0], t + 0);
    c = _addcarry_u64(c, t[1], pv[1], t + 1);
    c = _addcarry_u64(c, t[2], pv[2], t + 2);
    c = _addcarry_u64(c, t[3], pv[3], t + 3);
    c = _addcarry_u64(c, t[4], pv[4], t + 4);
  }

}

// ------------------------------------------------

void Int::MontMult(Int *a, Int *b, Int *p) {

  uint64_t t[10];
  uint64_t ah,al,bh,bl;
  uint64_t c = 0;
  uint64_t r0,r1;
  uint64_t *av = a->bits64;
  uint64_t *bv = b->bits64;
  uint64_t *pv = p->bits64;

  // i = 0
  memset(t,0,10*8);
  al = av[0] & 0x7FFFFFFFFFFFFULL;
  ah = av[0] >> 51;
  bl = bv[0] & 0x7FFFFFFFFFFFFULL;
  bh = bv[0] >> 51;
  t[0] = al * bl;
  r0 = t[0];
  r1 = (t[0] >> 51) + al * bh + ah * bl;
  t[0] = (r1 << 13) | (r0 & 0x7FFFFFFFFFFFFULL);
  r0 = r1 >> 51;
  r1 = ah * bh + r0;
  t[1] = r1;
  t[2] = r1 >> 51;

  for (int i = 1; i < 5; i++) {

    al = av[i] & 0x7FFFFFFFFFFFFULL;
    ah = av[i] >> 51;
    bl = bv[i] & 0x7FFFFFFFFFFFFULL;
    bh = bv[i] >> 51;

    c = 0;
    r0 = al * bl + (t[i] & 0x7FFFFFFFFFFFFULL) + c;
    r1 = (r0 >> 51) + al * bh + ah * bl + (t[i] >> 51);
    t[i] = (r1 << 13) | (r0 & 0x7FFFFFFFFFFFFULL);
    r0 = r1 >> 51;
    r1 = ah * bh + (t[i+1] & 0x7FFFFFFFFFFFFULL) + r0;
    t[i+1] = r1;
    c = r1 >> 51;
    for (int j = i + 2; j < 10; j++) {
      t[j] += c;
      c = t[j] >> 64;
      t[j] &= 0xFFFFFFFFFFFFFFFFULL;
    }

  }

  REDUCE(t, p);

  memcpy(bits64, t, 5 * 8);

}

// ------------------------------------------------

void Int::ModMulK1(Int *a, Int *b, Int *p) {

  MontMult(a, b, p);
  Int t(this);
  t.MontMult(GetR(p), p);
  Set(&t);

}

// ------------------------------------------------

void Int::ModSquareK1(Int *a, Int *p) {

  ModMulK1(a, a, p);

}

// ------------------------------------------------

void Int::SetupField(Int *p) {

  // Empty

}