#ifndef BSR_HPP
#define BSR_HPP

#include <algorithm>
#include <memory>
template <typename data_type, typename index_type>
class bcsr_matrix_class
{
public:
  bcsr_matrix_class (
    index_type n_rows_arg,
    index_type n_cols_arg,
    index_type bs_arg,
    index_type nnzb_arg)
    : n_rows (n_rows_arg)
    , n_cols (n_cols_arg)
    , bs (bs_arg)
    , nnzb (nnzb_arg)
    , values (new data_type[nnzb * bs * bs])
    , columns (new index_type[nnzb])
    , row_ptr (new index_type[n_rows + 1])
  {
  }

  void transpose_blocks (data_type *new_values)
  {
    std::unique_ptr<data_type[]> buffer (new data_type[bs * bs]);

    for (index_type row = 0; row < n_rows; row++)
      {
        for (index_type block = row_ptr[row]; block < row_ptr[row + 1]; block++)
          {
            data_type *new_block_data = new_values + bs * bs * block;
            data_type *old_block_data = values.get () + bs * bs * block;
            std::copy_n (old_block_data, bs * bs, buffer.get ());

            for (unsigned int i = 0; i < bs; i++)
              for (unsigned int j = 0; j < bs; j++)
                new_block_data[j * bs + i] = buffer[i * bs + j];
          }
      }
  }

  index_type size () const
  {
    return nnzb * bs * bs;
  }

  data_type *get_block_data (index_type row, index_type block_in_row)
  {
    return values.get() + (row_ptr[row] + block_in_row) * bs * bs;
  }

  data_type *get_block_data_by_column (index_type row, index_type column)
  {
    index_type block_in_row =
      std::distance (
        columns.get () + row_ptr[row],
        std::lower_bound (columns.get () + row_ptr[row], columns.get () + row_ptr[row + 1], column)
      );

    return get_block_data (row, block_in_row);
  }

public:
  const index_type n_rows {};
  const index_type n_cols {};

  const index_type bs {};
  const index_type nnzb {};

  const std::unique_ptr<data_type[]> values;
  const std::unique_ptr<index_type[]> columns;
  const std::unique_ptr<index_type[]> row_ptr;
};

#endif
