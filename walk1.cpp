#include <algorithm>
#include <iostream>
#include <gmpxx.h>
#include <math.h>
using std::cout;
using std::flush;
using std::sort;
using std::lower_bound;

#define R 128    // must be a power of 2
#define W 524288 // must be a power of 2
#define T 896
#define N 7168

struct tableentry { mpz_class x; mpz_class log; long long weight; };
tableentry table[N]; // of which T will be used in main computation

bool tablesort(tableentry i,tableentry j)
{
  return i.weight > j.weight; // move higher weights earlier
}
bool tablesort2(tableentry i,tableentry j)
{
  return i.x < j.x; // move smaller points earlier
}

mpz_class l("281474976710656");
mpz_class p("109058979322431746959182812013517394520037958891193115336877067190430268203759");

mpz_class power(const mpz_class &g,const mpz_class &e)
{
  mpz_class result;
  mpz_powm(result.get_mpz_t(),g.get_mpz_t(),e.get_mpz_t(),p.get_mpz_t());
  return result;
}

int hash(const mpz_class &w)
{
  return w.get_si() & (R-1);
}

int distinguished(const mpz_class &w)
{
  return !(w.get_si() & (W-1));
}

int main()
{
  gmp_randclass ra(gmp_randinit_default);

  mpz_class g = p / l; g = (g * g) % p;

  mpz_class s[R];
  mpz_class slog[R];
  for (int i = 0;i < R;++i) slog[i] = ra.get_z_bits(46) / W;
  for (int i = 0;i < R;++i) s[i] = power(g,slog[i]);

  long long totalnumsteps = 0;
  long long numsteps = 0;

  int tabledone = 0;
  while (tabledone < N) {
    mpz_class wlog = ra.get_z_bits(48);
    mpz_class w = power(g,wlog);
    for (int loop = 0;loop < 8*W;++loop) {
      if (distinguished(w)) {
        int i;
        for (i = 0;i < tabledone;++i) if (table[i].x == w) break;
        if (i < tabledone) {
          table[i].weight += 4*W + numsteps;
        } else {
          table[tabledone].x = w;
          table[tabledone].log = wlog;
          table[tabledone].weight = 4*W + numsteps;
          ++tabledone;
        }
        numsteps = 0;
        break;
      }
      int h = hash(w);
      wlog = wlog + slog[h];
      w = (w * s[h]) % p;
      ++numsteps;
      ++totalnumsteps;
    }
  }

  sort(table,table + N,tablesort);
  sort(table,table + T,tablesort2);

  cout << "alpha = " << W / sqrt(float(l.get_ui())/T) << "\n";
  cout << "r = " << R << "\n";
  cout << "W = " << W << "\n";
  cout << "T = " << T << "\n";
  cout << "N = " << N << "\n";
  cout << totalnumsteps << " precomputation steps; ";
  cout << totalnumsteps / sqrt(float(l.get_ui())*T) << "\n";
  totalnumsteps = 0;

  mpz_class hlog = ra.get_z_bits(48);
  mpz_class h = power(g,hlog);
  long long numsuccesses = 0;
  long long experiments = 0;

  for (;;) {
    numsteps = 0;
  
    mpz_class wdist = ra.get_z_bits(40);
    mpz_class w = (h * power(g,wdist)) % p;
    for (int loop = 0;loop < 8*W;++loop) {
      if (distinguished(w)) {
        tableentry desired;
        desired.x = w;
        tableentry *position = lower_bound(table,table + T,desired,tablesort2);
        if (position < table + T)
          if (position->x == w) {
            wdist = position->log - wdist;
          }
        break;
      }
      int h = hash(w);
      wdist = wdist + slog[h];
      w = (w * s[h]) % p;
      ++numsteps;
      ++totalnumsteps;
    }
    if (power(g,wdist) == h) {
      ++numsuccesses;
      hlog = ra.get_z_bits(48);
      h = power(g,hlog);
    }
    ++experiments;
    if (numsuccesses > 0 && !(experiments & (experiments - 1))) {
      cout << experiments << " experiments, "
           << totalnumsteps << " steps, "
           << numsuccesses << " successes, "
           << totalnumsteps/numsuccesses << " steps/success; "
           << (totalnumsteps/numsuccesses) / sqrt(float(l.get_ui())/T) << "\n";
      cout << flush;
    }
  }

  return 0;
}
