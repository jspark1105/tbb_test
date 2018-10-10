#include "TwistedHyperCube.h"

#include <array>
#include <cassert>
#include <cmath>

using namespace std;

HyperCube::HyperCube(int num_nodes, int shift /*=0*/)
  : num_nodes_(num_nodes), shift_(shift)
{
}

int HyperCube::NumNodes() const {
  return num_nodes_;
}

int HyperCube::Dimension() const {
  int dim = (int)round(log2(NumNodes()));
  assert(NumNodes() == (1 << dim));
  return dim;
}

int HyperCube::Peer(int id, int dim) const {
  assert(id >= 0 && id < NumNodes());
  return UnshiftedId(ShiftedId(id) ^ (1 << dim));
}

int HyperCube::RecvLocation(int id, int dim) const {
  assert(dim >= 0 && dim < Dimension());

  int recv_offset = ShiftedId(id) & (1 << dim);
  int base = (dim == Dimension() - 1) ? 0 : RecvLocation(id, dim + 1);
  return base + recv_offset;
}

int HyperCube::SendLocation(int id, int dim) const {
  return RecvLocation(Peer(id, dim), dim);
}

int HyperCube::ShiftedId(int id, int shift) const {
  int mask = (1 << Dimension()) - 1;
  return ((id << shift) | (id >> (-shift + Dimension()))) & mask;
}

int HyperCube::ShiftedId(int id) const {
  return ShiftedId(id, shift_);
}

int HyperCube::UnshiftedId(int id) const {
  return ShiftedId(id, Dimension() - shift_);
}

static const array<int, 8> normal_hcube_to_twisted_hcube_id = {
  0, 1, 3, 2, 4, 5, 7, 6,
};
static const array<int, 8>
  twisted_hcube_to_normal_hcube_id = normal_hcube_to_twisted_hcube_id;

TwistedHyperCube::TwistedHyperCube(int num_nodes, int shift /*=0*/)
  : HyperCube(num_nodes, shift), shift_(shift) {
  assert(num_nodes <= 8);
}

int TwistedHyperCube::Peer(int id, int dim) const {
  int shifted_dim = (dim - shift_ + Dimension()) % Dimension();
  int peer = id ^ ((1 << (shifted_dim + 1)) - 1);

  int ret = peer;
  if (id == 2 && peer == 5) {
    ret = 4;
  }
  else if (id == 3 && peer == 4) {
    ret = 5;
  }
  else if (id == 4 && peer == 3) {
    ret = 2;
  }
  else if (id == 5 && peer == 2) {
    ret = 3;
  }

  return ret;
}

int TwistedHyperCube::RecvLocation(int id, int dim) const {
  assert(dim >= 0 && dim < Dimension());

  int shifted_dim = (dim - shift_ + Dimension()) % Dimension();
  int recv_offset = 0;
  if (NumNodes() == 8 && shift_ > 0 && dim >= 1) {
    // special case because of the twist
    if (dim == 2) {
      if (id >= 2 && id <= 5) {
        recv_offset = 4;
      }
    }
    else if (dim == 1) {
      if (id == 1 || id == 2 || id == 4 || id == 6) {
        recv_offset = 2;
      }
    }
  }
  else if (shifted_dim == 0) {
    // special case because of the twist
    if ((id & 0x03) == 0x01 || (id & 0x03) == 0x02) {
      recv_offset = 1 << shift_;
    }
  }
  else {
    if (shifted_dim + shift_ >= Dimension()) {
      recv_offset = (id & (1 << shifted_dim)) >> (Dimension() - shift_);
    }
    else {
      recv_offset = (id & (1 << shifted_dim)) << shift_;
    }
  }
  int base = (dim == Dimension() - 1) ? 0 : RecvLocation(id, dim + 1);
  return base + recv_offset;
}

int TwistedHyperCube::SendLocation(int id, int dim) const {
  return RecvLocation(Peer(id, dim), dim);
}
