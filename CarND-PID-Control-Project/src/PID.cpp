#include "PID.h"
#include <limits>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
  prev_cte = std::numeric_limits<double>::max();
  int_cte = 0.0;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  if(prev_cte == std::numeric_limits<double>::max())
  {
    prev_cte = cte;
  }
  p_error = -Kp * cte;
  d_error = -Kd * (cte -prev_cte);
  i_error = -Ki * (int_cte+cte);
  prev_cte = cte;
  int_cte += cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */

  return p_error + i_error + d_error;  // TODO: Add your total error calc here!
}