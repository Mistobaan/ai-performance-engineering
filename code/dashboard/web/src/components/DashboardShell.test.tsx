import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DashboardShell } from '@/components/DashboardShell';

const push = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push }),
  usePathname: () => '/',
}));

describe('DashboardShell navigation', () => {
  beforeEach(() => {
    push.mockReset();
  });

  it('renders the contracts tab', () => {
    render(
      <DashboardShell title="Test Dashboard">
        <div>content</div>
      </DashboardShell>
    );

    expect(screen.getByRole('link', { name: /contracts/i })).toBeInTheDocument();
  });

  it('routes to the contracts tab on 0 shortcut', async () => {
    const user = userEvent.setup();

    render(
      <DashboardShell title="Test Dashboard">
        <div>content</div>
      </DashboardShell>
    );

    await user.keyboard('0');

    expect(push).toHaveBeenCalledWith('/contracts');
  });
});
