
#include "matrix_converter.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <cmath>



template <typename data_type>
size_t csr_matrix_class<data_type>::get_matrix_size () const
{
  return nnz;
}

matrix_rows_statistic get_rows_statistics (
    unsigned int rows_count,
    const unsigned int *row_ptr)
{
  matrix_rows_statistic statistic {};
  statistic.min_elements_in_rows = std::numeric_limits<unsigned int>::max () - 1;

  unsigned int sum_elements_in_rows = 0;
  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];

    if (elements_in_row > statistic.max_elements_in_rows)
      statistic.max_elements_in_rows = elements_in_row;

    if (elements_in_row < statistic.min_elements_in_rows)
      statistic.min_elements_in_rows = elements_in_row;

    sum_elements_in_rows += elements_in_row;
  }

  statistic.avg_elements_in_rows = sum_elements_in_rows / rows_count;
  statistic.elements_in_rows_std_deviation = 0.0;

  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];
    statistic.elements_in_rows_std_deviation += std::pow (static_cast<double> (elements_in_row) - statistic.avg_elements_in_rows, 2);
  }
  statistic.elements_in_rows_std_deviation = std::sqrt (statistic.elements_in_rows_std_deviation / rows_count);

  return statistic;
}

template <typename data_type>
size_t ell_matrix_class<data_type>::estimate_size (csr_matrix_class<data_type> &matrix)
{
  const auto row_ptr = matrix.row_ptr.get ();
    matrix_rows_statistic statistic=get_rows_statistics (matrix.n, row_ptr);
    auto min_elements= statistic.min_elements_in_rows;
    auto max_elements= statistic.max_elements_in_rows;
    auto avg_elements=statistic.avg_elements_in_rows;
    auto std_deviation= statistic.elements_in_rows_std_deviation;
  //auto [min_elements, max_elements, avg_elements, std_deviation] = get_rows_statistics (matrix.n, row_ptr);
  size_t elements_in_rows = max_elements;

  return elements_in_rows * matrix.n;
}

template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix)
{
  const auto rows_count=matrix.n;
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();
  matrix_rows_statistic statistic=get_rows_statistics (rows_count, row_ptr);
  auto min_elements= statistic.min_elements_in_rows;
  auto max_elements= statistic.max_elements_in_rows;
  auto avg_elements=statistic.avg_elements_in_rows;
  auto std_deviation= statistic.elements_in_rows_std_deviation;
  elements_in_rows = max_elements;

  std::cout << "ELL: " << elements_in_rows
            << " elements in rows (min: " << min_elements
            << "; avg: " << avg_elements<< ")" << std::endl;

  const size_t elements_count = elements_in_rows * rows_count;
  data.reset (new data_type[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[row + (element - start) *rows_count] = matrix.data[element];
      columns[row + (element - start) * rows_count] = col_ptr[element];
    }
  }
}

template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg)
{   rows_count=matrix.n;
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();
  elements_in_rows=elements_in_row_arg;

  const unsigned int elements_count = get_matrix_size ();

  data.reset (new data_type[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    /// Skip extra elements
    for (auto element = start; element < std::min (start + elements_in_row_arg, end); element++)
    {
      data[row + (element - start) * rows_count] = matrix.data[element];
      columns[row + (element - start) * rows_count] = col_ptr[element];
    }
  }
}

template <typename data_type>
size_t ell_matrix_class<data_type>::get_matrix_size () const
{

    return rows_count * elements_in_rows;
}

template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class(csr_matrix_class<data_type> &matrix)
{
  const auto non_zero_count=matrix.nnz;
  const auto rows_count=matrix.n;
  data.reset (new data_type[non_zero_count]);
  cols.reset (new unsigned int[non_zero_count]);
  rows.reset (new unsigned int[non_zero_count]);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  unsigned int id = 0;
  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[id] = matrix.data[element];
      cols[id] = col_ptr[element];
      rows[id] = row;
      id++;
    }
  }
}

template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start)

{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();
  const auto rows_count=matrix.n;
  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
      if (element - start >= element_start)
        elements_count++;
  }

  data.reset (new data_type[get_matrix_size ()]);
  cols.reset (new unsigned int[get_matrix_size ()]);
  rows.reset (new unsigned int[get_matrix_size ()]);

  unsigned int id = 0;
  for (unsigned int row = 0; row < rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      if (element - start >= element_start)
      {
        data[id] = matrix.data[element];
        cols[id] = col_ptr[element];
        rows[id] = row;
        id++;
      }
    }
  }
}

template <typename data_type>
size_t coo_matrix_class<data_type>::get_matrix_size () const
{
  return elements_count;
}

template <typename data_type>
hybrid_matrix_class<data_type>::hybrid_matrix_class (csr_matrix_class<data_type> &matrix)

{
}

template <typename data_type>
void hybrid_matrix_class<data_type>::allocate(csr_matrix_class<data_type> &matrix, double percent)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto n=matrix.n;

  //auto [_1, max_elements, avg_elements, _2] = get_rows_statistics (n, row_ptr);
  matrix_rows_statistic statistic=get_rows_statistics (n, row_ptr);
  auto min_elements= statistic.min_elements_in_rows;
  auto max_elements= statistic.max_elements_in_rows;
  auto avg_elements=statistic.avg_elements_in_rows;
  auto std_deviation= statistic.elements_in_rows_std_deviation;
  const unsigned int elements_per_ell = avg_elements; //+ (max_elements - avg_elements) * percent;

  ell_matrix = std::make_unique<ell_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use more than avg elements in an ELL row
  coo_matrix = std::make_unique<coo_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use elements before avg elements in an COO row


  std::cout << "ELL elements per row: " << elements_per_ell << "; "
            << "COO elements: " << coo_matrix->get_matrix_size () << "; ELL elements: " << ell_matrix->get_matrix_size () << "; "
            << "COO/ELL Ratio: " << static_cast<double> (coo_matrix->get_matrix_size ()) / ell_matrix->get_matrix_size () << std::endl;

}

template class csr_matrix_class<float>;
template class csr_matrix_class<double>;

template class ell_matrix_class<float>;
template class ell_matrix_class<double>;

template class coo_matrix_class<float>;
template class coo_matrix_class<double>;

template class hybrid_matrix_class<float>;
template class hybrid_matrix_class<double>;
