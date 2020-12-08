
#ifndef MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H
#define MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H


#include <memory>

struct matrix_rows_statistic
{
  unsigned int min_elements_in_rows {};
  unsigned int max_elements_in_rows {};
  unsigned int avg_elements_in_rows {};
  double elements_in_rows_std_deviation {};
};

matrix_rows_statistic get_rows_statistics (
    unsigned int n,
    const unsigned int *row_ptr);

template <typename data_type>
class csr_matrix_class
{




  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;
  std::unique_ptr<unsigned int[]> row_ptr;
  unsigned int n;
  unsigned int nnz;
};

template <typename data_type>
class ell_matrix_class
{
public:
  ell_matrix_class () = delete;
  explicit ell_matrix_class (csr_matrix_class<data_type> &matrix);
  ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg);

  static size_t estimate_size (csr_matrix_class<data_type> &matrix);


  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;

  unsigned int elements_in_rows = 0;
  unsigned int rows_count=0;
};

template <typename data_type>
class coo_matrix_class
{
public:
  coo_matrix_class () = delete;
  explicit coo_matrix_class (csr_matrix_class<data_type> &matrix);
  coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start);


  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> cols;
  std::unique_ptr<unsigned int[]> rows;

private:
  size_t elements_count {};
};

template <typename data_type>
class hybrid_matrix_class
{
public:
  hybrid_matrix_class () = delete;
  explicit hybrid_matrix_class (csr_matrix_class<data_type> &matrix);

  void allocate (csr_matrix_class<data_type> &matrix, double percent);


  std::unique_ptr<ell_matrix_class<data_type>> ell_matrix;
  std::unique_ptr<coo_matrix_class<data_type>> coo_matrix;
};

#endif // MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H
