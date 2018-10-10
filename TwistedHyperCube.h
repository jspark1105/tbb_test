#pragma once

class HyperCube
{
 public:
  // Bits of node id can be shifted by the optional parameter.
  HyperCube(int num_nodes, int shift = 0);

  int NumNodes() const;
  int Dimension() const;
  virtual int Peer(int id, int dim) const;

  // send and recv location in recursive halving
  // for recursive doubling, pass Dimension() - 1 - dim
  virtual int RecvLocation(int id, int dim) const;
  virtual int SendLocation(int id, int dim) const;

 protected:
  int ShiftedId(int id, int shift) const;
  int ShiftedId(int id) const;
  int UnshiftedId(int id) const;

  int num_nodes_;
  int shift_;
};

// 0 - 1
// |   |
// 3 - 2
//   X
// 4 - 5
// |   |
// 7 - 6
class TwistedHyperCube : public HyperCube
{
 public:
  TwistedHyperCube(int num_nodes, int shift = 0);

  int Peer(int id, int dim) const override;

  // send and recv location in recursive halving
  // for recursive doubling, pass Dimension() - 1 - dim
  int RecvLocation(int id, int dim) const override;
  int SendLocation(int id, int dim) const override;

 private:
  int shift_;
};
