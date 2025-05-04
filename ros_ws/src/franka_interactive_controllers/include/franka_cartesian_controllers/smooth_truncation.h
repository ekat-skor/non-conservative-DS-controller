
#ifndef SMOOTH_TRUNCATION_H
#define SMOOTH_TRUNCATION_H

#include "passive_ds_typedefs.h"

/**
 * Implements C2-continuous quintic "smooth step" functions as in Appendix A of the paper.
 * H_{a,b}(x) = 0             for x <= a
 *             = 6t^5-15t^4+10t^3   for a < x < b, t = (x-a)/(b-a)
 *             = 1             for x >= b
 * and its complement 1 - H_{a,b}(x).
 */

inline realtype smooth_step(realtype x, realtype a, realtype b) {
  assert(b > a);
  if (x <= a) {
    return 0.0;
  } else if (x >= b) {
    return 1.0;
  } else {
    realtype t = (x - a) / (b - a);
    // quintic polynomial: 6t^5 - 15t^4 + 10t^3
    return ((6.0 * t - 15.0) * t + 10.0) * t * t * t;
  }
}

inline realtype smooth_rise(realtype x, realtype lo, realtype hi) {
  // rising edge from 0@lo to 1@hi
  return smooth_step(x, lo, hi);
}

inline realtype smooth_fall(realtype x, realtype lo, realtype hi) {
  // falling edge: 1@lo down to 0@hi
  return 1.0 - smooth_step(x, lo, hi);
}

// single-variable "bump" between [l0->l1] up and [h1->h0] down
class SmoothRiseFall {
private:
  realtype lo_zero_, lo_one_, hi_one_, hi_zero_;
public:
  SmoothRiseFall(realtype l0, realtype l1, realtype h1, realtype h0)
    : lo_zero_(l0), lo_one_(l1), hi_one_(h1), hi_zero_(h0) {}

  realtype operator()(realtype x) const {
    realtype r = smooth_rise(x, lo_zero_, lo_one_);
    realtype f = smooth_fall(x, hi_one_, hi_zero_);
    return r * f;
  }
};

// 2D bump: zero in two corner regions, 1 elsewhere, smooth transitions
class SmoothRiseFall2d {
private:
  realtype xlo_, xhi_, ylo_, yhi_, dx_, dy_;
public:
  SmoothRiseFall2d(realtype xlo, realtype xhi, realtype dx,
                   realtype ylo, realtype yhi, realtype dy)
    : xlo_(xlo), xhi_(xhi), ylo_(ylo), yhi_(yhi), dx_(dx), dy_(dy) {}

  realtype operator()(realtype x, realtype y) const {
    // region near lower-left corner
    realtype e1 = smooth_rise(x, xlo_ - dx_, xlo_);
    e1 *= smooth_fall(y, ylo_, ylo_ + dy_);
    // region near upper-right corner
    realtype e2 = smooth_fall(x, xhi_, xhi_ + dx_);
    e2 *= smooth_rise(y, yhi_ - dy_, yhi_);
    return 1.0 - (e1 + e2);
  }
};

// 2D hinge: zero in one corner region, 1 elsewhere
class SmoothRise2d {
private:
  realtype xlo_, dx_, ylo_, dy_;
public:
  SmoothRise2d(realtype xlo, realtype dx, realtype ylo, realtype dy)
    : xlo_(xlo), dx_(dx), ylo_(ylo), dy_(dy) {}

  realtype operator()(realtype x, realtype y) const {
    realtype r = smooth_rise(x, xlo_, xlo_ + dx_);
    r *= smooth_fall(y, ylo_, ylo_ + dy_);
    return 1.0 - r;
  }
};

#endif // SMOOTH_TRUNCATION_H







// DIFFERENT SMOOTHING FUNCTION 

// #ifndef SMOOTH_TRUNCATION_H
// #define SMOOTH_TRUNCATION_H

// #include "passive_ds_typedefs.h"



// // smooth step function returnning exactly 0.0 for val<lo and exactly 1.0 for val>1 and smooth transition in between
// inline realtype smooth_rise(realtype val,realtype lo, realtype hi){
//     assert(hi>=lo);
//     if(val>=hi)
//         return 1.0;
//     else if(val<lo)
//         return 0.0;
//     else{
//         realtype T = hi-lo;
//         T *= 2;
//         return 0.5+0.5*sin(2*M_PI*(val-lo)/T - M_PI*0.5);
//     }
// }


// // smooth step function returnning exactly 1.0 for val<hi and exactly 0.0 for val>lo and smooth transition in between
// inline realtype smooth_fall(realtype val,realtype hi, realtype lo){
//     assert(lo>=hi);
//     if(val>=lo)
//         return 0.0;
//     else if(val<hi)
//         return 1.0;
//     else{
//         realtype T = lo-hi;
//         T *= 2;
//         return 0.5+0.5*sin(2*M_PI*(val-lo)/T - M_PI*0.5);
//     }
// }


// // functor returning 0 for val<lo_zero_ and val>hi_zero_ , 1 for lo_one_<val<hi_one_ and again 0 for val>hi_zero_
// class SmoothRiseFall{
// private:
//     realtype lo_zero_;
//     realtype lo_one_;
//     realtype hi_one_;
//     realtype hi_zero_;

// public:
//     SmoothRiseFall(realtype l0,realtype l1,realtype h1,realtype h0):
//         lo_zero_(l0), lo_one_(l1), hi_one_(h1), hi_zero_(h0)
//     {}
//     realtype operator ()(realtype val) {
//         realtype h = smooth_rise(val,lo_zero_,lo_one_);
//         h *= smooth_fall(val,hi_one_,hi_zero_);
//         return h;
//     }
// };


// // functor returning 0 for a smooth function that is zero for all x,y such that x<x_high and y>yhigh
// //and zero for all x,y, such that x>x_low and y<_ylow
// //% and 1 elsewhere. The transition is smooth and the output goes exactly to
// //% zero at the specified boundaires. dx and dy contorl smoothness, large
// //% value =>  smooth result
// class SmoothRiseFall2d{
// private:
//     realtype xlo_;
//     realtype xhi_;
//     realtype ylo_;
//     realtype yhi_;
//     realtype dx_;
//     realtype dy_;

// public:
//     SmoothRiseFall2d(realtype xlo,realtype xhi,realtype dx,realtype ylo,realtype yhi,realtype dy):
//         xlo_(xlo), xhi_(xhi), dx_(dx), ylo_(ylo), yhi_(yhi), dy_(dy) {}
//     realtype operator()(realtype x,realtype y){
//         realtype h1 = smooth_rise(x,xlo_-dx_,xlo_);
//         h1 *= smooth_fall(y,ylo_,ylo_+dy_);
//         realtype h2 = smooth_fall(x,xhi_,xhi_+dx_);
//         h2 *= smooth_rise(y,yhi_-dy_,yhi_);
//         return 1.0-h1-h2;
//     }

// };


// // functor returning exactly 0 for x >= xlo and y =<ylo and >0  elsewhere. Smoothness controlled by dx and dy
// class SmoothRise2d{
// private:
//     realtype xlo_,dx_,ylo_,dy_;
// public:
//     SmoothRise2d(realtype xlo, realtype dx, realtype ylo, realtype dy):
//         xlo_(xlo),dx_(dx),ylo_(ylo),dy_(dy) {}

//     realtype operator() (realtype x,realtype y){
//         realtype h1 = smooth_rise(x,xlo_,xlo_+dx_);
//         h1 *= smooth_fall(y,ylo_,ylo_+dy_);
//         return 1-h1;
//     }

// };

// #endif // SMOOTH_TRUNCATION_H